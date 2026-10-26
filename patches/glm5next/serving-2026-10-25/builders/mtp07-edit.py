import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a)); s = s.replace(a, b, 1)
# (d) remember this step's scheduler output on non-last stages
rep("""                    assert isinstance(hidden_states, IntermediateTensors)
                    self.kv_connector_output = kv_connector_output
                    return hidden_states
""", """                    assert isinstance(hidden_states, IntermediateTensors)
                    self.kv_connector_output = kv_connector_output
                    self._glm5_pp_sched = scheduler_output  # [claude-mtp-07]
                    return hidden_states
""")
# (c) non-last stages receive the spec-decode results
rep("""            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
""", """            if (  # [claude-mtp-07] pipeline + spec decode: next token, counts, drafts
                self.speculative_config is not None
                and get_pp_group().world_size > 1
                and not get_pp_group().is_last_rank
                and not self.broadcast_pp_output
            ):
                self._glm5_pp_receive_spec()
            elif self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
""")
# (a) last stage: no single-token broadcast when spec decode is on
rep("""            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(""",
"""            if (
                not self.broadcast_pp_output
                and pp.world_size > 1
                and pp.is_last_rank
                and self.speculative_config is None  # [claude-mtp-07] spec: sent after drafting
            ):
                self._pp_broadcast_prev_sampled_token_ids(""")
# (b) last stage: broadcast after drafting, before bookkeeping
rep("""        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
""", """        if (  # [claude-mtp-07]
            spec_config is not None
            and get_pp_group().world_size > 1
            and get_pp_group().is_last_rank
            and not self.broadcast_pp_output
        ):
            self._glm5_pp_broadcast_spec(sampler_output.sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
""")
# (f) earlier stages, async: trim the optimistic -1 placeholders exactly as the last stage does
rep("""                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
""", """                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                    # [claude-mtp-07b] async spec decode: output_token_ids was inflated by the optimistic
                    # extend above; align it with the scheduler as the last stage does (its elif below).
                    if num_output_tokens < len(req_state.output_token_ids):
                        del req_state.output_token_ids[num_output_tokens:]
                        if req_index is not None:
                            self.input_batch.num_tokens_no_spec[req_index] = (
                                self.input_batch.num_prompt_tokens[req_index]
                                + num_output_tokens
                            )
""")
# (e) the two helpers
methods = '''    def _glm5_pp_broadcast_spec(self, sampled_token_ids: torch.Tensor) -> None:
        """[claude-mtp-07] Last pipeline stage, spec decode: send every earlier stage, per request,
        [next input token, valid sampled count (position correction), accepted count (KDA/mamba state),
        draft tokens]. Without it the earlier stages never learn how many drafts were accepted: in
        async mode they cannot build the next step's inputs, and in either mode their KDA layers keep
        the default of one accepted token."""
        pp = get_pp_group()
        if self._is_all_reqs_chunked_prefill():
            return
        n = self.input_batch.num_reqs
        k = self.num_spec_tokens
        out = sampled_token_ids[:n]
        buf = torch.zeros((n, 3 + k), dtype=torch.int32, device=self.device)
        acc = (out != -1).sum(dim=1).to(torch.int32)
        buf[:, 2] = acc
        prev = self.input_batch.prev_sampled_token_ids
        vcnt = self.valid_sampled_token_count_gpu
        if self.use_async_scheduling and prev is not None and vcnt is not None:
            buf[:, 0] = prev[:n, 0].to(torch.int32)
            buf[:, 1] = vcnt[:n].to(torch.int32)
        else:
            idx = (acc.long() - 1).clamp(min=0)
            buf[:, 0] = out.gather(1, idx[:, None])[:, 0].to(torch.int32)
            buf[:, 1] = acc
        d = self._draft_token_ids
        if k and isinstance(d, torch.Tensor) and d.dim() == 2 and d.shape[0] >= n:
            kk = min(k, d.shape[1])
            buf[:, 3 : 3 + kk] = d[:n, :kk].to(torch.int32)
        torch.distributed.broadcast(buf, src=pp.rank, group=pp.device_group)

    def _glm5_pp_receive_spec(self) -> None:
        """[claude-mtp-07] Earlier pipeline stages, spec decode: receive what _glm5_pp_broadcast_spec
        sent and apply it the way the last stage applied it to itself."""
        pp = get_pp_group()
        assert not pp.is_last_rank
        n = self.input_batch.num_reqs
        k = self.num_spec_tokens
        buf = torch.zeros((n, 3 + k), dtype=torch.int32, device=self.device)
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(buf, src=pp.last_rank, group=pp.device_group)
        else:
            buf[:, 2] = 1
        sched = getattr(self, "_glm5_pp_sched", None)
        self._glm5_pp_sched = None
        if self.model_config.is_hybrid and sched is not None:
            ar = torch.arange(k + 1, device=self.device, dtype=torch.int32)[None, :]
            fake = torch.where(
                ar < buf[:, 2:3],
                torch.zeros((), dtype=torch.int32, device=self.device),
                torch.full((), -1, dtype=torch.int32, device=self.device),
            )
            self._update_states_after_model_execute(fake, sched)
        if not self.use_async_scheduling:
            return
        self._copy_valid_sampled_token_count(buf[:, 0].contiguous(), buf[:, 1].contiguous())
        if self.input_batch.prev_sampled_token_ids is None:
            self.input_batch.prev_sampled_token_ids = buf[:, 0:1].contiguous()
        self._draft_token_ids = (
            buf[:, 3:].to(self.input_ids.gpu.dtype).contiguous() if k else None
        )
        # Same bookkeeping as _pp_receive_prev_sampled_token_ids_to_input_batch.
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:n])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
            pos = self.input_batch.num_tokens_no_spec[i]
            self.input_batch.is_token_ids[i, pos] = True
            self.input_batch.num_tokens_no_spec[i] = pos + 1
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

'''
rep("""    @torch.inference_mode()
    def _glm5_profiler_tick(self) -> None:""", methods + """    @torch.inference_mode()
    def _glm5_profiler_tick(self) -> None:""")
open(p, "w").write(s); print("gpu_model_runner.py: pipeline spec-decode results sent to every stage (claude-mtp-07)")
