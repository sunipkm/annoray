//! The extension's allocator: the system allocator, with large blocks advised to use
//! transparent huge pages on Linux (as numpy does for its arrays).
//!
//! Large results are fresh allocations, and touching them page by page dominated large
//! elementwise operations: one fault per 4 KiB page. Advising huge pages (2 MiB) before the
//! memory is first touched lets the kernel map it in far fewer faults where transparent huge
//! pages are in `madvise` mode (the Ubuntu default). It is advice only: nothing changes
//! where the kernel declines, and nothing changes on other systems.

use std::alloc::{GlobalAlloc, Layout, System};

/// Blocks at least this large are advised.
#[cfg(target_os = "linux")]
const ADVISE_FROM: usize = 4 << 20;

/// The huge page size on x86-64 and aarch64 Linux.
#[cfg(target_os = "linux")]
const HUGE_PAGE: usize = 2 << 20;

/// The system allocator, advising huge pages for large blocks.
pub(crate) struct Allocator;

// SAFETY: every method forwards to `System` with the caller's arguments and returns its
// result unchanged; `advise` only gives the kernel advice about pages inside the block just
// returned (it never reads, writes or frees memory).
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds `GlobalAlloc::alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        advise(p, layout.size());
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as for `alloc`.
        let p = unsafe { System.alloc_zeroed(layout) };
        advise(p, layout.size());
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, i.e. from `System`, with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as for `dealloc`; the caller upholds `GlobalAlloc::realloc`'s contract.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        advise(p, new_size);
        p
    }
}

/// Advise huge pages for the whole 2 MiB pages inside `[p, p + len)`.
#[cfg(target_os = "linux")]
fn advise(p: *mut u8, len: usize) {
    if p.is_null() || len < ADVISE_FROM {
        return;
    }
    let addr = p.addr();
    let (Some(start), Some(end)) = (
        addr.checked_next_multiple_of(HUGE_PAGE),
        addr.checked_add(len)
            .and_then(|e| e.checked_rem(HUGE_PAGE).and_then(|r| e.checked_sub(r))),
    ) else {
        return;
    };
    if end > start {
        // SAFETY: `[start, end)` lies inside the block just allocated (page-aligned, as
        // `madvise` requires). MADV_HUGEPAGE is advice: it does not change the memory's
        // contents or validity, and failure (e.g. THP disabled) is harmless and ignored.
        #[allow(unsafe_code)]
        unsafe {
            libc::madvise(
                p.with_addr(start).cast(),
                end.saturating_sub(start),
                libc::MADV_HUGEPAGE,
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
const fn advise(_: *mut u8, _: usize) {}
