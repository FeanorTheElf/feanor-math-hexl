// recursion limit is required by macro `cpp!`
#![recursion_limit = "256"]

#![feature(test)]
#![feature(allocator_api)]

#![doc = include_str!("../Readme.md")]

extern crate cpp;
extern crate feanor_math;
extern crate test;

pub mod hexl;
pub mod conv;