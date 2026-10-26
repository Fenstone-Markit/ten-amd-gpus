import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a)); s = s.replace(a, b, 1)
# 1. keep the batch queue (prompt chunks still overlap across stages); mark the mode instead of forcing size 1
rep("""            logger.info("[claude-mtp-04] speculative decoding without async scheduling: batch queue %d -> 1",
                        self.batch_queue_size)
            self.batch_queue_size = 1
""", """            logger.info("[claude-mtp-08] speculative decoding without async scheduling: batch queue %d kept for "
                        "prompts; decode steps run one at a time and drafts are fetched only when the queue is empty",
                        self.batch_queue_size)
            self._glm5_spec_sync_pp = True
""")
# 2. hold new batches while any request is decoding and a batch is in flight
rep("""        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())""",
"""        model_executed = False
        deferred_scheduler_output = None
        _glm5_hold = (  # [claude-mtp-08]
            getattr(self, "_glm5_spec_sync_pp", False)
            and len(batch_queue) > 0
            and any(r.num_computed_tokens >= r.num_prompt_tokens for r in self.scheduler.running)
        )
        if self.scheduler.has_requests() and not _glm5_hold:
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())""")
# 3. fetch drafts after the step's output is recorded, once nothing else is in flight
rep("""        self._attach_iteration_details(engine_core_outputs, iteration_details)

        # NOTE(nick): We can either handle the deferred tasks here or save""",
"""        self._attach_iteration_details(engine_core_outputs, iteration_details)
        if getattr(self, "_glm5_spec_sync_pp", False) and not batch_queue:  # [claude-mtp-08]
            _draft_ids = self.model_executor.take_draft_token_ids()
            if _draft_ids is not None:
                self.scheduler.update_draft_token_ids(_draft_ids)

        # NOTE(nick): We can either handle the deferred tasks here or save""")
# 4. post_step does not fetch in that mode
rep("""        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            _t0 = time.perf_counter()  # [claude-mtp-06]""",
"""        if (
            self.check_for_draft_tokens
            and not self.async_scheduling
            and model_executed
            and not getattr(self, "_glm5_spec_sync_pp", False)  # [claude-mtp-08]
        ):
            _t0 = time.perf_counter()  # [claude-mtp-06]""")
open(p, "w").write(s); print("core.py: batch queue kept for prompts; decode one step at a time; drafts after output (claude-mtp-08)")
