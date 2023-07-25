extern crate blas_src;

use ndarray::ViewRepr;
use ndarray_linalg::Norm;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::ndarray::prelude::*;
use numpy::*;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;

fn rs_argmax(vector: ArrayBase<ViewRepr<&f64>, Ix1>) -> usize {
    let mut argmax: usize = 0;
    for i in 0..vector.len() {
        if vector[i].abs() > vector[argmax].abs() {
            argmax = i;
        };
    }
    return argmax;
}

#[pyfunction]
fn argmax<'a>(py: Python<'a>, vector: PyReadonlyArray1<f64>) -> usize {
    let vector = vector.as_array();
    return rs_argmax(ArrayView::from(vector));
}

#[pyfunction]
fn matmul<'a>(
    py: Python<'a>,
    array_a: PyReadonlyArray2<f64>,
    array_b: PyReadonlyArray2<f64>,
) -> &'a PyArray<f64, Ix2> {
    return (array_a.as_array().dot(&(array_b.as_array()))).into_pyarray(py);
}

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
        argmax = rs_argmax(ArrayView::from(&t_vector));

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
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    Ok(())
}
