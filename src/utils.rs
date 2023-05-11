use std::fmt::{Debug, Display, Write};
use std::ops::{Bound, Index, RangeBounds};
use std::{fs, iter};

use ndarray::ArrayView2;

use crate::F;

pub trait Step<T> {
    type Iter: Iterator<Item = T>;
    fn step_size(&self, step_size: T) -> Self::Iter;
}

impl<R: RangeBounds<F>> Step<F> for R {
    type Iter = Box<dyn Iterator<Item = F>>;

    fn step_size(&self, step_size: F) -> Self::Iter {
        let mut start = match self.start_bound() {
            Bound::Included(f) => *f,
            Bound::Excluded(f) => f + step_size,
            Bound::Unbounded => F::MIN,
        };
        assert!(!start.is_nan(), "not a number");
        let end = self.end_bound().cloned();
        Box::new(iter::once(start).chain(iter::from_fn(move || {
            if start == f64::INFINITY && matches!(end, Bound::Unbounded) {
                Some(start)
            } else {
                let next = start + step_size;
                #[allow(clippy::float_cmp)]
                {
                    assert_ne!(start, next, "step size too small, did not increase float");
                }
                start = next;
                match end {
                    // TODO check if this is a reasonable implementation of approx_eq
                    Bound::Included(max) if (max - start).abs() < 0.00001 => Some(start),
                    Bound::Included(max) | Bound::Excluded(max) if max > start => Some(start),
                    _ => None,
                }
            }
        })))
    }
}

pub fn min<'a>(iter: impl IntoIterator<Item = &'a F>) -> F {
    iter.into_iter().copied().reduce(F::min).unwrap()
}
pub fn max<'a>(iter: impl IntoIterator<Item = &'a F>) -> F {
    iter.into_iter().copied().reduce(F::max).unwrap()
}
// pub fn max_i<'a>(iter: impl IntoIterator<Item = &'a F>) -> usize {
//     let mut max = F::MIN;
//     let mut max_i = 0;
//     for (idx, &value) in iter.into_iter().enumerate() {
//         if value > max {
//             max_i = idx;
//             max = value;
//         }
//     }
//     #[allow(clippy::float_cmp)]
//     {
//         assert_ne!(max, F::MIN, "Expect to find a non F::MIN element");
//     }
//     max_i
// }

pub fn sort_i_dec<
    I: Index<usize, Output = F> + IntoIterator<IntoIter = II> + Copy,
    II: ExactSizeIterator,
>(
    list: I,
) -> Vec<usize> {
    let mut indices = (0..list.into_iter().len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&a, &b| list[b].total_cmp(&list[a]));
    indices
}

#[allow(unused)]
pub fn csv_dump(array: ArrayView2<impl Display>, filename: &str) {
    let mut s = String::new();
    for row in array.rows() {
        for col in row {
            write!(s, "{col},").unwrap();
        }
        writeln!(s);
    }
    fs::write(filename.to_string() + ".csv", s).unwrap();
}

#[allow(unused)]
pub fn csv_dump_2d(
    array: impl IntoIterator<Item = impl IntoIterator<Item = impl Debug>>,
    filename: &str,
) {
    let mut s = String::new();
    for row in array {
        for col in row {
            write!(s, "{col:?},").unwrap();
        }
        writeln!(s);
    }
    fs::write(format!("{filename}.csv"), s).unwrap();
}

#[allow(unused)]
pub fn csv_dump_1d(array: impl IntoIterator<Item = impl Debug>, filename: &str) {
    let mut s = String::new();
    for row in array {
        writeln!(s, "{row:?}");
    }
    fs::write(format!("{filename}.csv"), s).unwrap();
}
