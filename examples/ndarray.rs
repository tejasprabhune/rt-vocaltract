extern crate ndarray;
extern crate ndarray_npy;

use ndarray::s;

fn main() {
    let a = ndarray::array![[1, 2, 3], [4, 5, 6]];

    println!("{:?}", a);

    let venture: ndarray::Array2<f64> = ndarray_npy::read_npy("resources/venture.npy").unwrap();
    println!("{:?}", venture.shape());

    // ORDER OF ARTICULATORS:
    // LI XY, UL XY, LL XY, TT XY, TB XY, TD XY
    println!("{:?}", venture.slice(s![.., 0..2]));
}
