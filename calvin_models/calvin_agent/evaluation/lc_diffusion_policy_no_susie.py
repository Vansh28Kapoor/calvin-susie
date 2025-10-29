import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_dataset import CalvinDataset
import jax
from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
import numpy as np
import os
import orbax.checkpoint

class LCDiffusionPolicyNoSUSIE:
    """Language-Conditioned Diffusion Policy for CALVIN evaluation WITHOUT SUSIE goal images.
    
    This policy uses the lc_ddpm_bc agent (language-conditioned DDPM BC)
    which directly predicts actions from language instructions and observations
    WITHOUT using concatenated goal images.
    """
    
    def __init__(self):
        # We need to first create a dataset object to supply to the agent
        # Use original mini_dataset (not susie-processed version)
        train_paths = [[
            "mini_dataset/0.tfrecord",
            "mini_dataset/1.tfrecord"
        ]]

        dataset_kwargs = dict(
            shuffle_buffer_size=5000,  # Reduced from 25000
            prefetch_num_batches=5,    # Reduced from 20
            augment=True,
            augment_next_obs_goal_differently=False,
            augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.1],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            goal_relabeling_strategy="delta_goals",
            goal_relabeling_kwargs=dict(goal_delta=[0, 20]),
            load_language=True,
            skip_unlabeled=True,
            relabel_actions=False,
            act_pred_horizon=4,
            obs_horizon=1,
            load_susie_goal_images=False,  # This is the key difference - no SUSIE goal images
        )

        ACT_MEAN = [
            2.9842544e-04,
            -2.6099570e-04,
            -1.5863389e-04,
            5.8916201e-05,
            -4.4560504e-05,
            8.2349771e-04,
            9.4075650e-02,
        ]

        ACT_STD = [
            0.27278143,
            0.23548537,
            0.2196189,
            0.15881406,
            0.17537235,
            0.27875036,
            1.0049515,
        ]

        PROPRIO_MEAN = [ # We don't actually use proprio so we're using dummy values for this
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        PROPRIO_STD = [ # We don't actually use proprio so we're using dummy values for this
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": ACT_MEAN,
                "std": ACT_STD,
                "min": ACT_MEAN,
                "max": ACT_STD
            },
            "proprio": {
                "mean": PROPRIO_MEAN,
                "std": PROPRIO_STD,
                "min": PROPRIO_MEAN,
                "max": PROPRIO_STD
            }
        }

        action_metadata = {
            "mean": ACT_MEAN,
            "std": ACT_STD,
        }

        train_data = CalvinDataset(
            train_paths,
            42,
            batch_size=64,  # Reduced from 256 to reduce memory usage
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            **dataset_kwargs,
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        # Initialize text processor for language conditioning
        self.text_processor = text_processors["muse_embedding"]()
        
        # The dataset returns raw language strings in goals, but the agent expects embeddings
        # Convert the language strings to embeddings for agent initialization
        if "goals" in example_batch and "language" in example_batch["goals"]:
            # Extract language strings and convert to embeddings
            lang_strings = example_batch["goals"]["language"]
            # lang_strings can be bytes or arrays of bytes (e.g. shape (B,1)).
            # Normalize to a flat list of Python strings before encoding.
            flat_lang_strings = []
            for s in lang_strings:
                if isinstance(s, bytes):
                    flat_lang_strings.append(s.decode("utf-8"))
                elif isinstance(s, (list, tuple)) and len(s) == 1 and isinstance(s[0], bytes):
                    flat_lang_strings.append(s[0].decode("utf-8"))
                elif hasattr(s, 'decode') and not isinstance(s, str):
                    # e.g. numpy scalar bytes
                    try:
                        flat_lang_strings.append(s.decode('utf-8'))
                    except Exception:
                        flat_lang_strings.append(str(s))
                else:
                    flat_lang_strings.append(str(s))

            # Encode to embeddings
            lang_embeddings = self.text_processor.encode(flat_lang_strings)
            # Ensure embeddings are numpy array of shape (B, E) and dtype float32
            lang_embeddings = np.asarray(lang_embeddings)
            
            # Remove all singleton dimensions except the batch dimension
            # Target: (B, E) where B is batch size, E=512
            while lang_embeddings.ndim > 2:
                # Find singleton dimensions (excluding first dim which is batch)
                singleton_dims = [i for i in range(1, lang_embeddings.ndim) if lang_embeddings.shape[i] == 1]
                if singleton_dims:
                    lang_embeddings = np.squeeze(lang_embeddings, axis=singleton_dims[0])
                else:
                    break
            
            # If 1D, expand to (1, E)
            if lang_embeddings.ndim == 1:
                lang_embeddings = lang_embeddings[np.newaxis, ...]
            
            lang_embeddings = lang_embeddings.astype(np.float32)

            # Replace raw strings with embeddings
            example_batch["goals"]["language"] = lang_embeddings

        # Next let's initialize the agent - this is the key difference for lc_ddpm_bc
        agent_kwargs = dict(
            score_network_kwargs=dict(
                time_dim=32,
                num_blocks=3,
                dropout_rate=0.1,
                hidden_dim=256,
                use_layer_norm=True,
            ),
            language_conditioned=True,  # Key difference: language conditioned
            early_goal_concat=None,      # Not used for language conditioning
            shared_goal_encoder=None,    # Not used for language conditioning
            use_proprio=False,
            beta_schedule="cosine",
            diffusion_steps=20,
            action_samples=1,
            repeat_last_step=0,
            learning_rate=3e-4,
            warmup_steps=2000,
            actor_decay_steps=int(2e6),
        )

        # Use the standard ResNet encoder (not FiLM since we don't need goal image conditioning)
        encoder_def = encoders["resnetv1-34-bridge"](
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish"
        )
        
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        
        # Create the lc_ddpm_bc agent (which is really gc_ddpm_bc with language_conditioned=True)
        agent = agents["gc_ddpm_bc"].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **agent_kwargs,
        )

        print("Loading checkpoint...") 
        resume_path = os.getenv("LC_POLICY_CHECKPOINT_NO_SUSIE")
        if resume_path is None:
            # Fallback to regular checkpoint if no specific no-susie checkpoint is provided
            resume_path = os.getenv("LC_POLICY_CHECKPOINT")
        if resume_path is None:
            raise ValueError("Neither LC_POLICY_CHECKPOINT_NO_SUSIE nor LC_POLICY_CHECKPOINT environment variable is set")
        
        # Check if the path points to a directory with checkpoint subdirectories
        # If so, find the actual checkpoint directory (e.g., checkpoint_500000)
        if os.path.isdir(resume_path):
            checkpoint_dirs = [d for d in os.listdir(resume_path) if d.startswith('checkpoint_')]
            if checkpoint_dirs:
                # Use the checkpoint with the highest number
                checkpoint_dirs.sort(key=lambda x: int(x.split('_')[1]))
                resume_path = os.path.join(resume_path, checkpoint_dirs[-1])
                print(f"Using checkpoint: {resume_path}")
            
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=agent)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        agent = restored

        # save the loaded agent
        self.agent = agent
        self.action_statistics = action_metadata

        # Prepare action buffer for temporal ensembling
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=bool)

    def reset(self):
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=bool)

    def predict_action(self, image_obs: np.ndarray, language_instruction: str):
        """Predict action from observation and language instruction (NO goal image concatenation).
        
        Args:
            image_obs: RGB image observation (H, W, 3) - just the observation, no goal concatenation
            language_instruction: Natural language instruction string
            
        Returns:
            action: Predicted 7-DOF action (x, y, z, rx, ry, rz, gripper)
        """
        # Use the observation directly without any goal image concatenation
        # This assumes the agent was trained with 3-channel inputs (standard RGB)
        assert image_obs.shape[-1] == 3, f"Expected 3 channels (RGB), got {image_obs.shape[-1]}"
        
        # Process the language instruction
        language_embedding = self.text_processor.encode([language_instruction])
        language_embedding = np.asarray(language_embedding)
        language_embedding = np.squeeze(language_embedding)
        
        # Sample actions from the language-conditioned policy with observation only
        action = self.agent.sample_actions(
            {"image": image_obs[np.newaxis, ...]}, 
            {"language": language_embedding}, 
            seed=jax.random.PRNGKey(42), 
            temperature=0.0, 
        )
        action = np.array(action.tolist())

        # Shift action buffer for temporal ensembling
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * np.array([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False]
        ], dtype=bool)

        # Add to action buffer
        self.action_buffer[0] = action
        self.action_buffer_mask[0] = np.array([True, True, True, True], dtype=bool)
        
        # Ensemble temporally to predict action
        action_prediction = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0
        ) / np.sum(self.action_buffer_mask[:, 0], axis=0)

        # Make gripper action either -1 or 1
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        return action_prediction