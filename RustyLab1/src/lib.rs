extern crate blas_src;

use ndarray_rand::rand_distr::Uniform;
use numpy::ndarray::prelude::*;
use numpy::*;
use numpy::{IntoPyArray, PyArray};
use ndarray_rand::RandomExt;
use pyo3::prelude::*;
use ndarray_linalg::Norm;


/// Formats the sum of two numbers as string.
#[pyfunction]
fn power<'a>(
    py: Python<'a>,
    array_A: PyReadonlyArray2<f64>,
    tolerance: f64,
) -> (&'a PyArray<f64, Ix1>, f64) {
    let array_A = array_A.as_array();
    let mut eigenvalue = 0.0;
    let mut old_eigenvalue = 0.0;
    let mut eigenvector = Array::random(array_A.nrows(), Uniform::new(0.0, 1.0));
    let mut t_vector: Array<f64, Ix1>;
    let mut is_converged = false;
    let mut argmax: usize = 0;

    while !is_converged {
        t_vector = array_A.dot(&eigenvector);

        /* find argmax */
        for i in 0..t_vector.len() {
            if t_vector[i].abs() > t_vector[argmax].abs() {
                argmax = i
            };
        }

        eigenvalue = t_vector.norm_l2();
        eigenvector = t_vector / eigenvalue;
        if (old_eigenvalue - eigenvalue).abs() / eigenvalue.abs() > tolerance {
            old_eigenvalue = eigenvalue;
        } else {
            is_converged = true;
        }
    }

    if array_A.dot(&eigenvector)[argmax] / eigenvector[argmax] < 0.0 {
        eigenvalue = eigenvalue * -1.0
    }

    return (eigenvector.into_pyarray(py), eigenvalue);
}

/// A Python module implemented in Rust.
#[pymodule]
fn RustyLab1(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(power, m)?)?;
    Ok(())
}