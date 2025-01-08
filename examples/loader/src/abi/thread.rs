use axlog::info;
use arceos_posix_api::{self as api, ctypes};
use api::{sys_pthread_create, sys_pthread_exit, sys_pthread_join, sys_pthread_self};
use core::{ffi::c_void, mem};

use crate::load::EXEC_ZONE_START;

#[unsafe(no_mangle)]
pub extern "C" fn abi_pthread_create(
    res: *mut ctypes::pthread_t,
    attr: *const ctypes::pthread_attr_t,
    start_routine: extern "C" fn(arg: *mut c_void) -> *mut c_void,
    arg: *mut c_void,   // void *__restrict
) -> i32 {
    info!("[ABI:Thread] Create a new thread!");

    info!("res: {:p}", res);
    info!("attr: {:p}", attr);
    info!("start_routine: {:p}", start_routine);
    info!("arg: {:p}", arg);

    let adjusted_start_routine = unsafe {
        mem::transmute::<usize, extern "C" fn(arg: *mut c_void) -> *mut c_void>(
            (start_routine as usize) + EXEC_ZONE_START
        )
    };

    unsafe {
        sys_pthread_create(
            res, 
            attr, 
            adjusted_start_routine, 
            arg
        )
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn abi_pthread_join(
    thread: ctypes::pthread_t, 
    retval: *mut *mut c_void
) -> i32 {
    info!("[ABI:Thread] Wait for the given thread to exit!");
    unsafe {
        sys_pthread_join(thread, retval)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn abi_pthread_exit(retval: *mut c_void) -> ! {
    info!("[ABI:Thread] Exit the current thread!");
    sys_pthread_exit(retval);
}

#[unsafe(no_mangle)]
pub extern "C" fn abi_pthread_self() -> ctypes::pthread_t {
    info!("[ABI:Thread] Get the `pthread` struct of current thread!");
    sys_pthread_self()
}

pub fn abi_pthread_mutex_init() {
    info!("[ABI:Thread] Initialize a mutex!");
}

pub fn abi_pthread_mutex_lock() {
    info!("[ABI:Thread] Lock the given mutex!");
}

pub fn abi_pthread_mutex_unlock() {
    info!("[ABI:Thread] Unlock the given mutex!");
}