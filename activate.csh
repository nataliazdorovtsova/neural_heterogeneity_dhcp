# tcsh:  source activate.csh
set _proj = /imaging/astle/natalia/neural_heterogeneity_dhcp

setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cpu

if ( $?SLURM_CPUS_PER_TASK ) then
    set _nthreads = $SLURM_CPUS_PER_TASK
else
    set _nthreads = 4
endif
setenv OMP_NUM_THREADS $_nthreads
setenv MKL_NUM_THREADS $_nthreads
setenv OPENBLAS_NUM_THREADS $_nthreads
setenv NUMEXPR_NUM_THREADS $_nthreads
setenv XLA_FLAGS "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$_nthreads"

setenv PATH "${_proj}/env/bin:${PATH}"
if ( $?PYTHONPATH ) then
    setenv PYTHONPATH "${_proj}:${PYTHONPATH}"
else
    setenv PYTHONPATH "${_proj}"
endif

echo "neural_heterogeneity_dhcp — ${_nthreads} threads, x64 on"
unset _proj _nthreads
