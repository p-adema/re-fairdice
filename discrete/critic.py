from flax import nnx
import jax.numpy as jnp
from policy import MLP, CNNEncoder

class Critic(nnx.Module):
    def __init__(self, 
                 observation_dim,
                 action_dim: int = 0, 
                 hidden_dims = [256, 256], 
                 activation = nnx.relu, 
                 layer_norm: bool = False,
                 state_action_input: bool = False,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        if state_action_input:
            observation_dim += action_dim
        self.critic = MLP(
            observation_dim, hidden_dims=hidden_dims, activation=activation, rngs = rngs, layer_norm=layer_norm
        )

    def __call__(self, obesrvation):
        return self.critic(obesrvation)


class CNNCritic(nnx.Module):
    """
    Critic network with CNN encoder for image observations.
    
    Used for environments like minecart-rgb-v0 where observations are images.
    The CNN encoder is the same architecture as used in CNNDiscretePolicy
    to ensure consistent feature extraction.
    """
    def __init__(self,
                 feature_dim: int = 256,
                 hidden_dims: list = [256, 256],
                 activation = nnx.relu,
                 layer_norm: bool = False,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        """
        Initialize CNN Critic.
        
        Args:
            feature_dim: Output dimension of CNN encoder
            hidden_dims: Hidden layer dimensions for MLP after encoder
            activation: Activation function
            layer_norm: Whether to use layer normalization
            rngs: Random number generators
        """
        # CNN encoder (same architecture as policy for consistency)
        self.encoder = CNNEncoder(feature_dim=feature_dim, rngs=rngs)
        
        # MLP layers after CNN encoder, outputs scalar value
        self.critic_mlp = MLP(
            din=feature_dim,
            dout=1,  # Scalar value output
            hidden_dims=hidden_dims,
            activation=activation,
            rngs=rngs,
            layer_norm=layer_norm
        )
        
        self.feature_dim = feature_dim
        
    def __call__(self, images):
        """
        Compute value estimate from image observations.
        
        Args:
            images: Image tensor of shape (batch, height, width, channels)
                   Expected to be normalized to [0, 1]
                   
        Returns:
            Value estimates of shape (batch, 1)
        """
        # Encode images to features
        features = self.encoder(images)
        
        # Compute value
        return self.critic_mlp(features)