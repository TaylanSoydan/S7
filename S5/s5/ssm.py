from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    
    Identity = np.ones_like(Lambda) # New
    #Identity = np.ones(Lambda.shape[0]) 
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)

def apply_ssm_new(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional, input_dependent=True):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
            input_dependent (bool):  whether the output computation depends on C_tilde
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    #Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],Lambda_bar.shape[0]))
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],Lambda_bar.shape[1]))
    if input_dependent:
        Bu_elements = jax.vmap(lambda B_barr,u: B_barr @ u)(B_bar,input_sequence)
    else:
        Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    #Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        if input_dependent:
            return jax.vmap(lambda C_tilde, x: 2*(C_tilde @ x).real)(C_tilde, xs)
        else:
            return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        if input_dependent:
            return jax.vmap(lambda C_tilde, x: (C_tilde @ x).real)(C_tilde, xs)
        else:
            return jax.vmap(lambda x: (C_tilde @ x).real)(xs)

import math
def compute_inv_dt(key, features, dt_min, dt_max):
    # Generate random values
    rand_values = jax.random.uniform(key, (features,))
    
    # Compute dt
    dt = np.exp(
        rand_values * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = np.clip(dt, a_min=1e-4)

    # Compute inverse of softplus
    inv_dt = dt + np.log(-np.expm1(-dt))

    return inv_dt

def weight_init(minval, maxval):
    def init(key, shape, dtype=np.float32):
        return jax.random.uniform(key, shape, dtype, minval, maxval)
    return init

def bias_init(dt_min, dt_max):
    def init(key, shape, dtype=np.float32):
        return compute_inv_dt(key, shape[0], dt_min, dt_max)
    return init
from typing import Any, Callable, Sequence
class SimpleDense(nn.Module):
    features: int
    kernel_init: Callable
    bias_init: Callable
    name: str = None

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(f'{self.name}_kernel',
                            self.kernel_init,  # Initialization function
                            (inputs.shape[-1], self.features))  # Shape info.
        y = np.dot(inputs, kernel)
        bias = self.param(f'{self.name}_bias', self.bias_init, (self.features,))
        y = y + bias
        return y

class S5SSM(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    input_dependent: bool = True
    stablessm_a: bool = True

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            P = 2*self.P
        else:
            P = self.P

        
        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im


        if self.input_dependent:  
            self.dt_rank = math.ceil(self.H/16)
            if self.bidirectional:
                print("Bidirectional Model")
                self.x_proj = nn.Dense(self.dt_rank + P * 2 * self.H * 2 + P * self.H * 2, name = "x_proj")
            else:
                self.x_proj = nn.Dense(self.dt_rank + P * self.H * 2 + P * self.H * 2, name = "x_proj") #d,b,c
            dt_init_std = self.dt_rank**-0.5 * self.step_rescale
            #weight_min, weight_max = -dt_init_std, dt_init_std
            key = jax.random.PRNGKey(0)
            kernel_initializer = weight_init(-dt_init_std,dt_init_std)
            bias_initializer = bias_init(self.dt_min, self.dt_max)
            self.step_proj = SimpleDense(features=P, 
                        kernel_init=kernel_initializer,
                        bias_init=bias_initializer, name = "step_proj")
            

        else:

            # Initialize input to state (B) matrix
            B_init = lecun_normal()
            B_shape = (P, self.H)
            self.B = self.param("B",
                                lambda rng, shape: init_VinvB(B_init,
                                                            rng,
                                                            shape,
                                                            self.Vinv),
                                B_shape)
            B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

            # Initialize state to output (C) matrix
            if self.C_init in ["trunc_standard_normal"]:
                C_init = trunc_standard_normal
                C_shape = (self.H, P, 2)
            elif self.C_init in ["lecun_normal"]:
                C_init = lecun_normal()
                C_shape = (self.H, P, 2)
            elif self.C_init in ["complex_normal"]:
                C_init = normal(stddev=0.5 ** 0.5)
            else:
                raise NotImplementedError(
                    "C_init method {} not implemented".format(self.C_init))

            if self.C_init in ["complex_normal"]:
                if self.bidirectional:
                    C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                    self.C_tilde = C[..., 0] + 1j * C[..., 1]

                else:
                    C = self.param("C", C_init, (self.H, self.P, 2))
                    self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                if self.bidirectional:
                    self.C1 = self.param("C1",
                                        lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                        C_shape)
                    self.C2 = self.param("C2",
                                        lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                        C_shape)

                    C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                    C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                    self.C_tilde = np.concatenate((C1, C2), axis=-1)

                else:
                    self.C = self.param("C",
                                        lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                        C_shape)

                    self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])


        # Discretize
        if not self.input_dependent:
            if self.discretization in ["zoh"]:
                #if self.input_dependent:
                #    self.Lambda_bar, self.B_bar = jax.vmap(discretize_zoh)(self.Lambda, B_tilde, step)
                #else:
                self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
            elif self.discretization in ["bilinear"]:
                self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
            else:
                raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        
        if self.stablessm_a:
            a = 1
            b = 0.5
            Lambda = -np.sqrt((-1 - b * self.Lambda)/(a*self.Lambda))
        else:
            Lambda = self.Lambda

        if self.input_dependent:
            x_dbl = self.x_proj(input_sequence)
            step, B_projected, C_projected = np.split(x_dbl, indices_or_sections = [self.dt_rank,2*self.H*self.P+self.dt_rank],axis = -1)
            step = self.step_proj(step)
            B_reshaped = B_projected.reshape(-1, self.P, self.H, 2)
            
            if self.bidirectional:
                C_reshaped = C_projected.reshape(-1, self.H, self.P * 2, 2)
            else:
                C_reshaped = C_projected.reshape(-1, self.H, self.P, 2)
                
            step = jax.nn.softplus(step)
            
            B = B_reshaped[..., 0] + 1j * B_reshaped[..., 1]
            C = C_reshaped[..., 0] + 1j * C_reshaped[..., 1]

            def discretize_input_dependent(B_tilde, Delta):
                """ Discretize a diagonalized, continuous-time linear SSM
                    using zero-order hold method.
                    Args:
                        Lambda (complex64): diagonal state matrix              (P,)
                        B_tilde (complex64): input matrix                      (P, H)
                        Delta (float32): discretization step sizes             (P,)
                    Returns:
                        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
                """
                
                Identity = np.ones_like(Lambda) # New
                #Identity = np.ones(Lambda.shape[0]) 
                Lambda_bar = np.exp(Lambda * Delta)
                B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
                return Lambda_bar, B_bar

            Lambda_bar, B_bar = jax.vmap(discretize_input_dependent)(B, step)
            #self.Lambda_bar, self.B_bar = jax.vmap(discretize_zoh, in_axes=(None, 0, 0))(self.Lambda, B, step)

            ys = apply_ssm_new(Lambda_bar,
                       B, #self.B_bar,
                       C, #self.C_tilde,
                        input_sequence,
                        self.conj_sym,
                        self.bidirectional,
                        self.input_dependent)
        else:
            ys = apply_ssm(self.Lambda_bar,
            self.B_bar,
            self.C_tilde,
            input_sequence,
            self.conj_sym,
            self.bidirectional)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def init_S5SSM(H,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional,
               input_dependent,
               stablessm_a,
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional,
                   input_dependent=input_dependent,
                   stablessm_a=stablessm_a)
