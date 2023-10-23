pub type Cost = f64;

pub mod cost_factors {
    pub static HASH_COST_FACTOR: f64 = 1.4;
    pub static MEM_ALLOC_FACTOR: f64 = 2.;
    pub static PRED_COST_FACTOR: f64 = 0.25;
    pub static FILTER_COST_FACTOR: f64 = 0.1;
}

#[derive(Clone)]
pub struct CostEstimator {
    disk_read_factor: f64,
}

impl CostEstimator {
    /// The `num` argument should be the total number of elements (before taking selectivity into
    /// account). Note that this includes the cost of writing those read byes into memory as well.
    pub fn disk_read(&self, num: usize, elem_size: usize, selectivity: f64) -> Cost {
        if selectivity <= 0.2 {
            self.disk_read_factor * (num * elem_size) as f64 * selectivity
        } else {
            self.disk_read_factor * (num * elem_size) as f64
        }
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self {
            disk_read_factor: 20.,
        }
    }
}
