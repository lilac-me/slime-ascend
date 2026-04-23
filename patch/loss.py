import torch
from argparse import Namespace


# megatron_actor
# store_prefix = "ref_"
# def compute_log_prob(
#     self,
#     data_iterator: list[DataIterator],
#     num_microbatches: list[int],
#     store_prefix: str = "",
# ) -> dict[str, list[torch.Tensor]]:

#     with timer(f"{store_prefix}log_probs"):
#         return forward_only(
#             get_log_probs_and_entropy,
#             self.args,
#             self.model,
#             data_iterator,
#             num_microbatches,
#             store_prefix=store_prefix,
#         )
# megatron forward_backward_func call -> forward_step call -> get_log_probs_and_entropy


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    Computes on the **full** logits ``[T, V]`` tensor at once (instead of
    per-sample slicing) so backward traverses ``[T, V]`` only once, then
    extracts per-sample response portions.

    When ``entropy_coef == 0``, entropy is computed under ``torch.no_grad()``
    to avoid retaining the computation graph and to skip cloning.
    """
    assert non_loss_data
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1)) # [S, B, V] -> [T, V]

    # Apply rollout temperature scaling to logits to match rollout-time log-probs.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = parallel_state.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(
        T, device, unconcat_tokens, total_lengths, response_lengths, qkv_format, max_seq_lens, args.allgather_cp
    )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv: opsm or gspo
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False, chunk_size: int = -1):
    logits = logits.contiguous()
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)

            if with_entropy:
                entropys = []
                for logits_chunk in logits_chunks:
                    entropy_input = logits_chunk.clone()
                    entropys.append(compute_entropy_from_logits(entropy_input, tp_group))
                entropy = torch.cat(entropys, dim=0)

            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
        else:
            if with_entropy:
                entropy_input = logits.clone()
                entropy = compute_entropy_from_logits(entropy_input, tp_group)

            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)
