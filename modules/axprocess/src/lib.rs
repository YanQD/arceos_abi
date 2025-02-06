#![cfg_attr(not(test), no_std)]
#![feature(doc_auto_cfg)]

mod process;
mod stdio;
mod fd_manager;
mod api;
mod flags;
mod link;