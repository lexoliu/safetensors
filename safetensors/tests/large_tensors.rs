use safetensors::{
    slice::{slice_byte_ranges, TensorIndexer},
    tensor::{Metadata, TensorInfo, TensorView},
    Dtype, SafeTensorError,
};
use std::{num::NonZeroUsize, ops::Bound};

#[test]
fn large_tensor_metadata_fits_the_target_byte_address_space() {
    // A Qwen token embedding: 622 MB of BF16 data, over 2^32 bits.
    let info = TensorInfo {
        dtype: Dtype::BF16,
        shape: vec![151_936, 2048],
        data_offsets: (0, 622_329_856),
    };
    let metadata = Metadata::new(None, vec![("embedding".into(), info)]).unwrap();
    assert_eq!(metadata.data_len(), 622_329_856);
}

#[test]
fn large_tensor_view_checks_bytes_without_bit_count_overflow() {
    assert!(matches!(
        TensorView::new(Dtype::BF16, vec![151_936, 2048], &[]),
        Err(SafeTensorError::InvalidTensorView(..))
    ));
}

#[test]
fn large_tensor_slice_preserves_high_byte_offsets() {
    let (ranges, shape) = slice_byte_ranges(
        Dtype::BF16,
        &[151_936, 2048],
        &[
            TensorIndexer::Select(151_935),
            TensorIndexer::Narrow(
                Bound::Included(2044),
                Bound::Excluded(2048),
                NonZeroUsize::MIN,
            ),
        ],
    )
    .unwrap();
    assert_eq!(ranges, vec![(622_329_848, 622_329_856)]);
    assert_eq!(shape, vec![4]);
}
