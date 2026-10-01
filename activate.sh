# bash:  source activate.sh
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# float64 throughout. JAX defaults to float32, which is not good enough for
# Cholesky factors and log-determinants of a 270-dim covariance.
export JAX_ENABLE_X64=1

# No GPUs anywhere on this cluster — fail loudly rather than silently falling
# back, so a mis-set expectation surfaces immediately.
export JAX_PLATFORMS=cpu

# Respect the SLURM allocation when inside a job; be a good citizen otherwise.
_NTHREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$_NTHREADS"
export MKL_NUM_THREADS="$_NTHREADS"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$_NTHREADS"
unset _NTHREADS

export PATH="$PROJECT/env/bin:$PATH"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

echo "neural_heterogeneity_dhcp — python $("$PROJECT/env/bin/python" -V 2>&1 | cut -d' ' -f2), ${OMP_NUM_THREADS} threads, x64 on"
