from module_combination_mutation.utils import generate_input_ids_and_position_ids


def run_demo(
    batch_size: int = 2,
    vocab_size: int = 16,
    max_sequence_length: int = 8,
    is_random: bool = False,
) -> None:
    print("=" * 80)
    print(
        f"Run 10 times with batch_size={batch_size}, vocab_size={vocab_size}, "
        f"max_sequence_length={max_sequence_length}, is_random={is_random}"
    )
    print("=" * 80)

    for i in range(10):
        input_ids, position_ids, attention_mask = generate_input_ids_and_position_ids(
            batch_size=batch_size,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            is_random=is_random,
        )
        print(f"\n--- Call {i + 1} ---")
        print("input_ids:")
        print(input_ids)
        print("position_ids:")
        print(position_ids)
        print("attention_mask:")
        print(attention_mask)


if __name__ == "__main__":
    # 默认验证固定输出（每次调用都应一致）
    run_demo(is_random=False)

    # 如需验证随机输出，取消下一行注释
    # run_demo(is_random=True)
