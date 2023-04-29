use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use libax::task;

const NUM_TASKS: usize = 5;

static FINISHED_TASKS: AtomicUsize = AtomicUsize::new(0);

fn test1() {
    for i in 0..30 {
        println!("  tick {}", i);
        task::sleep(Duration::from_millis(500));
    }
    task::exit(0);
}

fn test2() {
    let sec = 1;
    let mut result: usize = 1;
    for i in 1..1000000 {
        result = result * i % 998244353;
    }
    println!("RESULT: {}", result);
    FINISHED_TASKS.fetch_add(1, Ordering::Relaxed);
    task::exit(0);
}

pub fn main() {
    println!("Hello, main task!");
    task::sleep(Duration::from_secs(1));
    println!("main task sleeped");

    // backgroud ticks, 0.5s x 30 = 15s
    task::spawn_fn(test1);

    // task n: sleep 3 x n (sec)
    for i in 0..NUM_TASKS {
        task::spawn_fn(test2);
    }

    while FINISHED_TASKS.load(Ordering::Relaxed) < NUM_TASKS {
        task::yield_now();
    }

    println!("Sleep tests run OK!");
}
