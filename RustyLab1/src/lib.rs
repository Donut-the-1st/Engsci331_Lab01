extern crate blas_src;

use ndarray::parallel::prelude::*;
use ndarray::ViewRepr;
use ndarray_linalg::Norm;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::ndarray::prelude::*;
use numpy::*;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;

fn par_mat_vec_mul(array_a: &ArrayView<f64, Ix2>, vector_x: &Array<f64, Ix1>) -> Array<f64, Ix1> {
    let mut vector_b = Array::zeros(vector_x.len());
    let a_iter = array_a.axis_chunks_iter(Axis(0), 32);
    let b_iter = vector_b.axis_chunks_iter_mut(Axis(0), 32);
    let zipped = a_iter.into_par_iter().zip(b_iter);

    zipped.for_each(|mut x| x.1.assign(&x.0.dot(vector_x)));

    return vector_b;
}

/*
fn par_mat_vec_mul(array_a: &ArrayView<f64, Ix2>, vector_x: &Array<f64, Ix1>) -> Array<f64, Ix1> {
    let mut vector_b= Array::zeros(vector_x.len());
    Zip::from(&mut vector_b)
        .and(array_a.rows())
        .par_for_each(|product, row| {*product = row.dot(vector_x)});
    return vector_b;
}
*/
fn rs_deflate(array: &mut ArrayViewMut<f64, Ix2>, eigenvector: &ArrayView<f64, Ix1>, eigenvalue: f64) {
    let scaled_eigenvector = eigenvalue * (eigenvector.clone().to_owned());
    array
        .axis_iter_mut(Axis(0))
        .zip(eigenvector.into_iter())
        .for_each(|(mut a, b)| a.assign(&(&a - *b * &scaled_eigenvector)));
}

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
fn argmax(vector: PyReadonlyArray1<f64>) -> usize {
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

fn power_sml_mat(array_A: ArrayView<f64, Ix2>, tolerance: f64) -> (Array<f64, Ix1>, f64) {
    let mut eigenvector = Array::random(array_A.nrows(), Uniform::new(0.0, 1.0));
    let mut eigenvalue = 0.0;
    let mut old_eigenvalue = 0.0;
    let mut is_converged = false;
    let mut argmax: usize = 0;

    while !is_converged {
        /* store t vector in eignenvector */
        eigenvector = array_A.dot(&eigenvector);
        /* find argmax */
        /* with t vector == eigenvector */
        argmax = rs_argmax(ArrayView::from(&eigenvector));
        eigenvalue = eigenvector.norm_l2();
        /* divide in place to convert t vector to eigenvector */
        eigenvector /= eigenvalue;
        if (old_eigenvalue - eigenvalue).abs() / eigenvalue.abs() > tolerance {
            old_eigenvalue = eigenvalue;
        } else {
            is_converged = true;
        }
    }

    if array_A.dot(&eigenvector)[argmax] / eigenvector[argmax] < 0.0 {
        eigenvalue = eigenvalue * -1.0;
    }

    return (eigenvector, eigenvalue);
}

fn power_lrg_mat(array_A: ArrayView<f64, Ix2>, tolerance: f64) -> (Array<f64, Ix1>, f64) {
    let mut eigenvector = Array::random(array_A.nrows(), Uniform::new(0.0, 1.0));
    let mut eigenvalue = 0.0;
    let mut old_eigenvalue = 0.0;
    let mut is_converged = false;
    let mut argmax: usize = 0;

    while !is_converged {
        /* store t vector in eignenvector */
        eigenvector = par_mat_vec_mul(&array_A, &eigenvector);
        /* find argmax */
        /* with t vector == eigenvector */
        argmax = rs_argmax(ArrayView::from(&eigenvector));
        eigenvalue = eigenvector.norm_l2();
        /* divide in place to convert t vector to eigenvector */
        eigenvector /= eigenvalue;
        if (old_eigenvalue - eigenvalue).abs() / eigenvalue.abs() > tolerance {
            old_eigenvalue = eigenvalue;
        } else {
            is_converged = true;
        }
    }

    if par_mat_vec_mul(&array_A, &eigenvector)[argmax] / eigenvector[argmax] < 0.0 {
        eigenvalue = eigenvalue * -1.0;
    }

    return (eigenvector, eigenvalue);
}

/// Power rule for 2D NumPy array, finds dominant eigenpair
#[pyfunction]
fn power<'py>(
    py: Python<'py>,
    array_A: PyReadonlyArray2<f64>,
    tolerance: f64,
) -> (&'py PyArray<f64, Ix1>, f64) {
    let array_A = array_A.as_array();
    let mut eigenvalue: f64 = 0.0;
    let mut eigenvector: Array<f64, Ix1> = Array::zeros(array_A.nrows());

    if eigenvector.len() < 255 {
        (eigenvector, eigenvalue) = power_sml_mat(array_A, tolerance);
    } else {
        (eigenvector, eigenvalue) = power_lrg_mat(array_A, tolerance);
    }

    return (eigenvector.into_pyarray(py), eigenvalue);
}
/// deflates an Array, A inplace

#[pyfunction]
fn deflate<'py>(mut array: PyReadwriteArray2<f64>, eigenvector: PyReadonlyArray1<f64>, eigenvalue: f64) {
    let mut array = array.as_array_mut();
    let eigenvector = eigenvector.as_array();
    rs_deflate(&mut array, &eigenvector, eigenvalue)
}


/// A Python module implemented in Rust.
#[pymodule]
fn RustyLab1(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deflate, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    Ok(())
}
