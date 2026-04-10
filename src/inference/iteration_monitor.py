"""Monitor hidden state evolution across loop iterations.

Hooks into the forward pass to record hidden states at the entry and exit
of the circuit block on each iteration, plus cosine similarities between
consecutive iterations.
"""
import torch
import torch.nn.functional as F


class IterationMonitor:
    """Records hidden states at circuit block boundaries during multi-pass execution.

    For a LayerDuplicator with block (i, j) and K iterations, the layer sequence is:
        [0..i-1] [i..j-1] [i..j-1] ... [i..j-1] [j..N-1]
                  pass 0    pass 1       pass K

    We hook the first layer of each pass (index i within the block) to capture
    the entry state, and the layer after the block exit to capture the exit state.
    """

    def __init__(self, block_i, block_j, iterations):
        self.block_i = block_i
        self.block_j = block_j
        self.iterations = iterations
        self.block_size = block_j - block_i

        # Recorded data per forward call (populated during forward pass)
        # Each element is a list of states for one forward call (one token)
        self.all_entry_states = []   # list of lists
        self.all_exit_states = []    # list of lists
        self._current_entries = []   # accumulator for current forward call
        self._current_exits = []
        self._hooks = []
        self._call_count = 0

    def _compute_hook_indices(self):
        """Compute which layer indices in the expanded sequence correspond to
        block entry and exit points."""
        # Total passes = 1 (original) + iterations (extra)
        total_passes = 1 + self.iterations

        entry_indices = []
        exit_indices = []

        for p in range(total_passes):
            # Entry: first layer of block on pass p
            entry_idx = self.block_i + p * self.block_size
            entry_indices.append(entry_idx)

            # Exit: layer after last layer of block on pass p
            # This is either the first layer of the next pass, or the first
            # decoder layer (block_j in original numbering)
            exit_idx = self.block_i + (p + 1) * self.block_size
            exit_indices.append(exit_idx)

        return entry_indices, exit_indices

    def attach(self, model):
        """Register forward hooks on the expanded model's layers."""
        self.all_entry_states = []
        self.all_exit_states = []
        self._current_entries = []
        self._current_exits = []
        self._hooks = []
        self._call_count = 0
        self._entry_count_per_call = 0

        layers = model.model.layers
        entry_indices, exit_indices = self._compute_hook_indices()
        total_passes = 1 + self.iterations

        for idx in entry_indices:
            if idx < len(layers):
                hook = layers[idx].register_forward_pre_hook(
                    self._make_entry_hook(total_passes)
                )
                self._hooks.append(hook)

        for idx in exit_indices:
            if idx < len(layers):
                hook = layers[idx].register_forward_pre_hook(
                    self._make_exit_hook(total_passes)
                )
                self._hooks.append(hook)

        return self

    def detach(self):
        """Remove all hooks and flush any pending data."""
        # Flush last call if it has data
        if self._current_entries:
            self.all_entry_states.append(self._current_entries)
            self._current_entries = []
        if self._current_exits:
            self.all_exit_states.append(self._current_exits)
            self._current_exits = []
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _make_entry_hook(self, total_passes):
        def hook(module, args):
            # Detect new forward call: when we've accumulated a full set of entries
            if len(self._current_entries) >= total_passes:
                self.all_entry_states.append(self._current_entries)
                self._current_entries = []
            hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            self._current_entries.append(hs.detach().clone())
        return hook

    def _make_exit_hook(self, total_passes):
        def hook(module, args):
            if len(self._current_exits) >= total_passes:
                self.all_exit_states.append(self._current_exits)
                self._current_exits = []
            hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            self._current_exits.append(hs.detach().clone())
        return hook

    @property
    def entry_states(self):
        """Entry states from the first forward call (prompt processing)."""
        if self.all_entry_states:
            return self.all_entry_states[0]
        return self._current_entries

    @property
    def exit_states(self):
        """Exit states from the first forward call (prompt processing)."""
        if self.all_exit_states:
            return self.all_exit_states[0]
        return self._current_exits

    def cosine_trajectory(self, call_idx=0):
        """Cosine similarity between consecutive exit states for a given forward call.

        Args:
            call_idx: Which forward call (0 = prompt processing, 1+ = token generation).

        Returns list of similarities: [sim(exit_0, exit_1), sim(exit_1, exit_2), ...].
        Each similarity is averaged across all sequence positions.
        """
        exits = self._get_exits(call_idx)
        sims = []
        for k in range(1, len(exits)):
            prev = exits[k - 1].float()
            curr = exits[k].float()
            cos_sim = F.cosine_similarity(prev, curr, dim=-1)
            sims.append(cos_sim.mean().item())
        return sims

    def entry_exit_similarities(self, call_idx=0):
        """Cosine similarity between entry and exit on each pass for a given forward call."""
        entries = self._get_entries(call_idx)
        exits = self._get_exits(call_idx)
        n = min(len(entries), len(exits))
        sims = []
        for k in range(n):
            entry = entries[k].float()
            ex = exits[k].float()
            cos_sim = F.cosine_similarity(entry, ex, dim=-1)
            sims.append(cos_sim.mean().item())
        return sims

    def _get_entries(self, call_idx):
        if call_idx < len(self.all_entry_states):
            return self.all_entry_states[call_idx]
        if call_idx == 0:
            return self._current_entries
        return []

    def _get_exits(self, call_idx):
        if call_idx < len(self.all_exit_states):
            return self.all_exit_states[call_idx]
        if call_idx == 0:
            return self._current_exits
        return []

    def summary(self):
        """Return a dict summarizing the iteration dynamics (prompt-only)."""
        trajectory = self.cosine_trajectory(call_idx=0)
        entry_exit = self.entry_exit_similarities(call_idx=0)
        return {
            "num_passes": 1 + self.iterations,
            "num_forward_calls": len(self.all_entry_states) + (1 if self._current_entries else 0),
            "exit_trajectory": trajectory,
            "entry_exit_sims": entry_exit,
            "final_similarity": trajectory[-1] if trajectory else None,
            "converged": trajectory[-1] > 0.999 if trajectory else False,
        }
