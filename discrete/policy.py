from flax import nnx
import jax.numpy as jnp
import jax.nn as jnn

class MuNetwork(nnx.Module):
    def __init__(self,
                 config):
        self.mu = nnx.Param(jnp.full((config.reward_dim,), 1.0))
        
    def __call__(self):
        return self.mu.value * 1.0

class MLP(nnx.Module):
    def __init__(self, din, dout = 1, hidden_dims = [256, 256], activation = nnx.relu, rngs: nnx.Rngs = nnx.Rngs(0), activate_final: bool = False, dropout_rate: float = 0.0, layer_norm: bool = False):
        dims = [din] + hidden_dims + [dout]
        
        layer = []
        for i in range(len(dims) - 1):
            layer.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))))
            if i < len(dims) - 2:
                if dropout_rate > 0:
                    layer.append(nnx.Dropout(dropout_rate, rngs=rngs))
                layer.append(activation)
                if layer_norm:
                    layer.append(nnx.LayerNorm(dims[i+1], rngs=rngs))
                
        if activate_final:
            layer.append(activation)
        
        self.layer = nnx.Sequential(
            *layer
        )
    
    def __call__(self, x):
        return self.layer(x)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
    
class DiscretePolicy(nnx.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 action_dim,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.mlp_layer = MLP(input_dim, hidden_dim, rngs=rngs)
        self.layer = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        

    def __call__(self, inputs):
        x = self.mlp_layer(inputs)
        
        logits = self.layer(x)
        # probs = nnx.softmax(logits, axis=-1)
        return tfd.Categorical(logits=logits)  # to match interface of GaussianPolicy


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class GaussianPolicy(nnx.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 action_dim,
                 activation = nnx.relu,
                 temperature=1.0,
                 log_std_scale: float = 1e-3,
                 tanh_squash_distribution=True,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 layer_norm: bool = False
                 ):
        
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution
        
        self.mlp_layer = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation=activation, rngs = rngs, activate_final=True, layer_norm=layer_norm)

        self.mean_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.std_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs = rngs, kernel_init=nnx.initializers.orthogonal(log_std_scale)
        )
        self.action_dim = action_dim

    def __call__(self, inputs):
        x = self.mlp_layer(inputs)
        
        means = self.mean_layer(x)
        if not self.tanh_squash_distribution:
            means = jnp.tanh(means)
        
        log_stds = self.std_layer(x)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
                
        dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * self.temperature)
        
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Tanh())
        else:
            return dist        
        
class MNDPolicy(nnx.Module):
    def __init__(self,
                 input_dim,
                 action_dim,
                 hidden_dims = [256, 256],
                 activation = nnx.relu,
                 n_mixture: int = 5,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 tanh_squash_distribution: bool = False,
                 temperature = 1.0,
                 log_std_scale: float = 1e-3,
                 layer_norm: bool = False
                 ):        
        self.temparature = temperature
        self.mlp_layer = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation=activation, rngs = rngs, activate_final=True, layer_norm=layer_norm)
        self.mean_layer = nnx.Linear(
            hidden_dims[-1], action_dim * n_mixture, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.std_layer = nnx.Linear(
            hidden_dims[-1], action_dim * n_mixture, rngs = rngs, kernel_init=nnx.initializers.orthogonal(log_std_scale)
        )
        self.logit_layer = nnx.Linear(
            hidden_dims[-1], n_mixture, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.tanh_squash = tanh_squash_distribution
        self.n_mixture = n_mixture
        self.action_dim = action_dim
    
    def __call__(self, observations):
        x = self.mlp_layer(observations)
        
        means = self.mean_layer(x).reshape(-1, self.n_mixture, self.action_dim)
        if not self.tanh_squash:
            means = nnx.tanh(means)
        log_stds = self.std_layer(x).reshape(-1, self.n_mixture, self.action_dim)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        logits = self.logit_layer(x)
        
        dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=logits),
        components_distribution=tfd.MultivariateNormalDiag(
        loc=means,       
        scale_diag=jnp.exp(log_stds) * self.temparature))
        
        if self.tanh_squash:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Tanh())
        else:
            return dist


class CNNEncoder(nnx.Module):
    def __init__(self, 
                 feature_dim: int = 256,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        # Input: (batch, 480, 480, 3)
        self.conv1 = nnx.Conv(
            in_features=3, out_features=32,
            kernel_size=(8, 8), strides=(4, 4),
            padding='VALID',
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )  # -> (batch, 118, 118, 32)
        
        self.conv2 = nnx.Conv(
            in_features=32, out_features=64,
            kernel_size=(4, 4), strides=(2, 2),
            padding='VALID',
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )  # -> (batch, 58, 58, 64)
        
        self.conv3 = nnx.Conv(
            in_features=64, out_features=64,
            kernel_size=(3, 3), strides=(2, 2),
            padding='VALID',
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )  # -> (batch, 28, 28, 64)
        
        self.conv4 = nnx.Conv(
            in_features=64, out_features=64,
            kernel_size=(3, 3), strides=(2, 2),
            padding='VALID',
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )  # -> (batch, 13, 13, 64)
        
        self.conv5 = nnx.Conv(
            in_features=64, out_features=64,
            kernel_size=(3, 3), strides=(2, 2),
            padding='VALID',
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )  # -> (batch, 6, 6, 64)
        
        self.fc = nnx.Linear(
            in_features=6 * 6 * 64,
            out_features=feature_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        
        self.feature_dim = feature_dim
        
    def __call__(self, x):
        """
        Process image observations.
        
        Args:
            x: Image tensor of shape (batch, height, width, channels)
               Expected to be normalized to [0, 1]
               
        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.conv4(x))
        x = nnx.relu(self.conv5(x))
        
        x = x.reshape(x.shape[0], -1)
        
        x = nnx.relu(self.fc(x))
        
        return x


class CNNDiscretePolicy(nnx.Module):
    def __init__(self,
                 action_dim: int,
                 feature_dim: int = 256,
                 hidden_dim: int = 256,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.encoder = CNNEncoder(feature_dim=feature_dim, rngs=rngs)
        
        self.mlp = MLP(
            din=feature_dim,
            dout=hidden_dim,
            hidden_dims=[],
            rngs=rngs,
            activate_final=True
        )
        
        self.action_head = nnx.Linear(
            in_features=hidden_dim,
            out_features=action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(0.01)
        )
        
        self.action_dim = action_dim
        
    def __call__(self, images):
        """        
        Args:
            images: Image tensor of shape (batch, height, width, channels)
            
        Returns:
            Categorical distribution over actions
        """
        features = self.encoder(images)
        
        x = self.mlp(features)
        
        logits = self.action_head(x)
        
        return tfd.Categorical(logits=logits)

    """
    Policy network for continuous actions with CNN encoder for image observations.
    
    Similar to GaussianPolicy but uses CNN encoder instead of MLP for observations.
    """
    def __init__(self,
                 action_dim: int,
                 feature_dim: int = 256,
                 hidden_dims: list = [256, 256],
                 temperature: float = 1.0,
                 tanh_squash_distribution: bool = True,
                 log_std_scale: float = 1e-3,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution
        
        self.encoder = CNNEncoder(feature_dim=feature_dim, rngs=rngs)
        
        self.mlp = MLP(
            din=feature_dim,
            dout=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            rngs=rngs,
            activate_final=True
        )
        
        # Mean and std heads
        self.mean_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.std_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(log_std_scale)
        )
        
        self.action_dim = action_dim
        
    def __call__(self, images):
        """
        Compute action distribution from image observations.
        
        Args:
            images: Image tensor of shape (batch, height, width, channels)
            
        Returns:
            MultivariateNormalDiag distribution over actions
        """
        # Encode images to features
        features = self.encoder(images)
        
        # Pass through MLP
        x = self.mlp(features)
        
        # Compute mean and log_std
        means = self.mean_layer(x)
        if not self.tanh_squash_distribution:
            means = jnp.tanh(means)
            
        log_stds = self.std_layer(x)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        
        # Create distribution
        dist = tfd.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * self.temperature
        )
        
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(
                distribution=dist,
                bijector=tfb.Tanh()
            )
        else:
            return dist