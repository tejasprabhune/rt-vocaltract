use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
    // The squared distance of the point to the origin.
    // pub fn norm(self) -> f64 {
    //     self.x * self.x + self.y * self.y
    // }
    // The squared distance to the other point given.
    // pub fn dist(self, rhs: Point) -> f64 {
    //     (self - rhs).norm()
    // }
}

/// To use interpolation we need to define the add operation with itself.
impl Add for Point {
    type Output = Point;
    fn add(self, rhs: Point) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

/// To calculate the distance of one point to another, we use subtraction.
impl Sub for Point {
    type Output = Point;
    fn sub(self, rhs: Point) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// To use interpolation we need to define the multiplication with a scalar.
impl Mul<f64> for Point {
    type Output = Point;
    fn mul(self, rhs: f64) -> Self::Output {
        Point {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

/// To use bezier or bsplines, we need to define a default.
impl Default for Point {
    fn default() -> Self {
        Point { x: 0.0, y: 0.0 }
    }
}

/// To use weights, we also need to define division with a scalar.
impl Div<f64> for Point {
    type Output = Point;
    fn div(self, rhs: f64) -> Self::Output {
        Point {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
