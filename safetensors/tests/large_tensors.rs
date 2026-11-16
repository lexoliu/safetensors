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

#[test]
fn unaddressable_shapes_and_empty_tensor_strides_stay_rejected() {
    for shape in [
        vec![2, usize::MAX],
        vec![0, 2, usize::MAX],
        vec![usize::MAX],
    ] {
        assert!(matches!(
            TensorView::new(Dtype::F32, shape.clone(), &[]),
            Err(SafeTensorError::ValidationOverflow)
        ));
        assert!(matches!(
            Metadata::new(
                None,
                vec![(
                    "invalid".into(),
                    TensorInfo {
                        dtype: Dtype::F32,
                        shape,
                        data_offsets: (0, 0),
                    },
                )],
            ),
            Err(SafeTensorError::ValidationOverflow)
        ));
    }
}

#[test]
fn packed_dtype_alignment_and_empty_shapes_are_preserved() {
    assert!(TensorView::new(Dtype::F4, vec![2], &[0]).is_ok());
    assert!(matches!(
        TensorView::new(Dtype::F4, vec![1], &[]),
        Err(SafeTensorError::MisalignedSlice)
    ));
    assert!(TensorView::new(Dtype::F4, vec![0, 1], &[]).is_ok());
    assert!(TensorView::new(Dtype::BF16, vec![0, 151_936, 2048], &[]).is_ok());
}
