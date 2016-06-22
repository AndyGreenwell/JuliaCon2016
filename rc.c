//rc.c

#include <julia.h>
#include <stdio.h>
#include <math.h>

#ifdef _OS_WINDOWS_
__declspec(dllexport) __cdecl
#endif

int r_f (jl_array_t *x, jl_array_t *f)
{
    double* xd = (double*)jl_array_data(x);
    double* fd = (double*)jl_array_data(f);

    fd[0] = pow(1.0 - xd[0], 2.0);
    fd[1] = 100.0 * pow(xd[1] - pow(xd[0], 2.0), 2.0);

    return 0;
}

int r_g (jl_array_t *x, jl_array_t * jac)
{
    double* xd = (double*)jl_array_data(x);
    double* jd = (double*)jl_array_data(jac);
    
    jd[0] = -2.0 * (1 - xd[0]);                         //1,1 element
    jd[1] = -400.0 * (xd[1] - pow(xd[0], 2.0)) * xd[0]; //2,1 element
    jd[2] = 0.0;                                        //1,2 element
    jd[3] = 200.0 * (xd[1] - pow(xd[0], 2.0));          //2,2 element

    return 0;
}

void r_fg (jl_array_t *x, jl_array_t *f, jl_array_t *jac)
{
    double* xd = (double*)jl_array_data(x);
    double* fd = (double*)jl_array_data(f);
    double* jd = (double*)jl_array_data(jac);
    
    fd[0] = pow(1.0 - xd[0], 2.0);
    fd[1] = 100.0 * pow(xd[1] - pow(xd[0], 2.0), 2.0);

    jd[0] = -2.0 * (1 - xd[0]);                         //1,1 element
    jd[1] = -400.0 * (xd[1] - pow(xd[0], 2.0)) * xd[0]; //2,1 element
    jd[2] = 0.0;                                        //1,2 element
    jd[3] = 200.0 * (xd[1] - pow(xd[0], 2.0));          //2,2 element
}

jl_value_t* yieldto(jl_value_t *solver, jl_value_t *vals) {
    jl_function_t *yieldto_func = jl_get_function(jl_base_module, "yieldto");
    return jl_call2(yieldto_func, solver, vals);
}

int istaskdone(jl_value_t *solver)
{
    jl_function_t *taskdone = jl_get_function(jl_base_module, "istaskdone");
    return jl_call1(taskdone, solver) == jl_true;
}

#define __noinline __attribute__((noinline))
void __noinline real_main()
{
    int n = 2;

    // Represent arrays that will be created and owned by C
    double *guess    = (double*)malloc(sizeof(double)*n);
    double *fval     = (double*)malloc(sizeof(double)*n);
    double *jacobian = (double*)calloc(n*n, sizeof(double));

    // Assign initial values into the above arrays
    guess[0] = -1.2;
    guess[1] =  1.0;
    fval[0]  =  1.0/0.0;
    fval[1]  =  1.0/0.0;

    // Root objects with the GC
    jl_value_t **args;
    // args[0] - NLsolve.DifferentiableMultivariateFunction(f!, g!, fg!)
    // args[1] - :f
    // args[2] - :g
    // args[3] - :fg
    // args[4] - array_type
    // args[5] - guess::Vector{Float64}
    // args[6] - fval::Vector{Float64}
    // args[7] - jacobian::Matrix{Float64}
    // args[8] - tuple function
    // args[9] - solver_task function
    
    JL_GC_PUSHARGS(args,10);

    jl_eval_string("using NLsolve");
    jl_eval_string("maintask = current_task()");
    jl_eval_string("f_t!(x::Vector, fvec::Vector) = yieldto(maintask, (:f, x, fvec))");
    jl_eval_string("g_t!(x::Vector, fjac::Matrix) = yieldto(maintask, (:g, x, fjac))");
    jl_eval_string("fg_t!(x::Vector, fvec::Vector, fjac::Matrix) = yieldto(maintask, (:fg, x, fvec, fjac))");
    
    args[0] = jl_eval_string("NLsolve.DifferentiableMultivariateFunction(f_t!, g_t!, fg_t!)");
    args[1] = (jl_value_t*) jl_symbol("f");
    args[2] = (jl_value_t*) jl_symbol("g");
    args[3] = (jl_value_t*) jl_symbol("fg");
    args[4] = (jl_value_t*) jl_apply_array_type(jl_float64_type, 1 );
    args[5] = (jl_value_t*) jl_ptr_to_array_1d(args[4], guess, n, 0);      // Wrap the C arrays as Julia arrays. The final "0" argument means
    args[6] = (jl_value_t*) jl_ptr_to_array_1d(args[4], fval, n, 0);       // that Julia does not take ownership of the array data for GC purposes.
    args[7] = (jl_value_t*) jl_ptr_to_array_1d(args[4], jacobian, n*n, 0);
    args[8] = (jl_value_t*) jl_get_function(jl_base_module, "tuple");
    args[9] = jl_eval_string("solver_task(df, guess) = @task nlsolve(df, guess)");

    { 
        jl_value_t* vals   = jl_call3((jl_function_t*) args[8], args[1], args[5], args[6]);  // Create the "vals" tuple
        jl_value_t* solver = jl_call2((jl_function_t*) args[9], args[0], args[5]);           // Create task for the solver
        
        JL_GC_PUSH2(&vals, &solver);

        jl_yield();
        while (!istaskdone(solver)) {
            if (jl_get_nth_field(vals, 0) == args[1]) {        // :f case
                args[5] = jl_get_nth_field(vals, 1);
                args[6] = jl_get_nth_field(vals, 2);
                r_f((jl_array_t*) args[5], (jl_array_t*) args[6]);
                vals = yieldto(solver, vals);
            } else if (jl_get_nth_field(vals, 0) == args[2]) { // :g case
                args[5] = jl_get_nth_field(vals, 1);
                args[7] = jl_get_nth_field(vals, 2);
                r_g((jl_array_t*) args[5], (jl_array_t*) args[7]); 
                vals = yieldto(solver, vals);
            } else if (jl_get_nth_field(vals, 0) == args[3]) { // :fg case
                args[5] = jl_get_nth_field(vals, 1);
                args[6] = jl_get_nth_field(vals, 2);
                args[7] = jl_get_nth_field(vals, 3);
                r_fg((jl_array_t*) args[5], (jl_array_t*) args[6], (jl_array_t*) args[7]); 
                vals = yieldto(solver, vals);
            }
        }
        jl_show(jl_stderr_obj(), vals);
        jl_eval_string("println(\"\n\")");
        JL_GC_POP();
    }
    JL_GC_POP();
    
    free(guess);
    free(fval);
    free(jacobian);
}

void __noinline real_main_wrapper()
{
    real_main();
}

int main()
{
    jl_init(NULL);

    real_main_wrapper(); // this should be a separate function from main()

    int ret = 0;
    jl_atexit_hook(ret);
    return ret;
}
