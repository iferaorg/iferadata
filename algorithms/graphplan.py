"""GraphPlan algorithm implemented as a class using PyTorch tensors."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import torch
import yaml


class GraphPlan:
    """GraphPlan planner.

    Parameters
    ----------
    filename:
        Path to the YAML planning problem description.
    device:
        Optional torch.device. If ``None`` the device is chosen based on CUDA
        availability.
    """

    def __init__(self, filename: str, device: Optional[torch.device] = None) -> None:
        self.filename = filename
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # These members are populated by the preprocessing step
        self.all_actions: List[Dict[str, List[str]]]
        self.preconds: torch.Tensor
        self.adds: torch.Tensor
        self.dels: torch.Tensor
        self.init_props: torch.Tensor
        self.goal_props: torch.Tensor
        self.num_props: int
        self.preconds_cpu: torch.Tensor
        self.adds_cpu: torch.Tensor

        self._preprocess_yaml()

    # ------------------------------------------------------------------
    # Helper methods for preprocessing
    # ------------------------------------------------------------------
    def _forward_pruning(
        self,
        preconds: torch.Tensor,
        adds: torch.Tensor,
        init_props: torch.Tensor,
        num_actions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pruning of unreachable propositions and actions."""

        prop_mask = init_props.clone()
        act_mask = torch.zeros(num_actions, dtype=torch.bool, device=self.device)
        changed = True
        while changed:
            changed = False
            new_acts = (~act_mask) & torch.all(preconds <= prop_mask[None, :], dim=1)
            if new_acts.any():
                act_mask |= new_acts
                changed = True
            new_props = (~prop_mask) & torch.any(adds[act_mask], dim=0)
            if new_props.any():
                prop_mask |= new_props
                changed = True
        return prop_mask, act_mask

    def _backward_pruning(
        self,
        preconds: torch.Tensor,
        adds: torch.Tensor,
        dels: torch.Tensor,
        goal_props: torch.Tensor,
        num_actions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform backward pruning to keep only relevant propositions/actions."""

        prop_mask = goal_props.clone()
        act_mask = torch.zeros(num_actions, dtype=torch.bool, device=self.device)
        changed = True
        while changed:
            changed = False
            new_acts = (~act_mask) & torch.any(adds & prop_mask[None, :], dim=1)
            if new_acts.any():
                act_mask |= new_acts
                changed = True
            new_props = (~prop_mask) & torch.any(preconds[act_mask], dim=0)
            if new_props.any():
                prop_mask |= new_props
                changed = True

        adds_rel = torch.any(adds[act_mask], dim=0)
        dels_rel = torch.any(dels[act_mask], dim=0)
        extra_props = (~prop_mask) & adds_rel & dels_rel
        if extra_props.any():
            prop_mask |= extra_props
            new_noops = (~act_mask) & torch.any(adds & extra_props[None, :], dim=1)
            if new_noops.any():
                act_mask |= new_noops
        return prop_mask, act_mask

    @staticmethod
    def _handle_variables(
        actions: List[Dict[str, List[str]]],
        propositions: List[str],
        prop_to_id: Dict[str, int],
        variables: Dict[str, Dict[str, object]],
    ) -> None:
        """Expand variable definitions into propositions and effects."""

        # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
        if not variables:
            return

        var_to_eqs: Dict[str, List[str]] = {}
        var_to_values: Dict[str, List[object]] = {}
        var_defs: Dict[str, Dict[str, object]] = {}

        for varname, vardef in variables.items():
            vtype = vardef["type"]
            var_defs[varname] = vardef
            if vtype == "enum":
                values = vardef["values"]
            elif vtype == "int":
                values = list(range(vardef["min_value"], vardef["max_value"] + 1))
            else:
                raise ValueError(f"Unknown variable type: {vtype}")
            var_to_values[varname] = values
            eqs = []
            for val in values:
                prop = f"{varname}_eq_{val}"
                if prop not in prop_to_id:
                    propositions.append(prop)
                    prop_to_id[prop] = len(propositions) - 1
                eqs.append(prop)
            var_to_eqs[varname] = eqs

        for act in actions:
            if "set" in act:
                seen_vars = set()
                for var, str_val in act["set"]:
                    if var in seen_vars:
                        raise ValueError(
                            f"Variable {var} set multiple times in action {act['name']}"
                        )
                    seen_vars.add(var)
                    if var not in var_to_eqs:
                        raise ValueError(f"Unknown variable {var}")
                    vardef = var_defs[var]
                    vtype = vardef["type"]
                    if vtype == "int":
                        val = int(str_val)
                    else:
                        val = str_val
                    eq = f"{var}_eq_{val}"
                    if eq not in prop_to_id:
                        raise ValueError(f"Invalid value {val} for variable {var}")
                    act["add"] = act.get("add", []) + [eq]
                    other_eqs = [e for e in var_to_eqs[var] if e != eq]
                    act["del"] = act.get("del", []) + other_eqs
                del act["set"]

            if "var_precond" in act:
                for expr in act["var_precond"]:
                    parts = expr.split()
                    if len(parts) != 3:
                        raise ValueError(f"Invalid var_precond expression: {expr}")
                    var, op, str_val = parts
                    if var not in var_to_eqs:
                        raise ValueError(f"Unknown variable {var}")
                    vardef = var_defs[var]
                    vtype = vardef["type"]
                    values = var_to_values[var]
                    if vtype == "int":
                        val = int(str_val)
                    else:
                        val = str_val
                    if op == "==":
                        eq = f"{var}_eq_{val}"
                        act["preconds"] = act.get("preconds", []) + [eq]
                    elif op == "!=":
                        eq = f"{var}_eq_{val}"
                        act["neg_preconds"] = act.get("neg_preconds", []) + [eq]
                    elif op in ["<=", ">="]:
                        if vtype != "int":
                            raise ValueError(
                                f"Inequality operators only supported for int variables: {expr}"
                            )
                        neg_eqs = []
                        for v in values:
                            if (op == "<=" and v > val) or (op == ">=" and v < val):
                                neg_eqs.append(f"{var}_eq_{v}")
                        act["neg_preconds"] = act.get("neg_preconds", []) + neg_eqs
                    else:
                        raise ValueError(f"Unknown operator in var_precond: {op}")
                del act["var_precond"]

    @staticmethod
    def _handle_negated_preconditions(
        actions: List[Dict[str, List[str]]],
        propositions: List[str],
        prop_to_id: Dict[str, int],
        initial: List[str],
    ) -> tuple[List[str], Dict[str, int], int, List[str]]:
        """Handle actions with negated preconditions by creating complementary props."""

        # pylint: disable=too-many-branches,too-many-statements
        for act in actions:
            if "neg_preconds" in act:
                for neg_p in act["neg_preconds"]:
                    if neg_p not in prop_to_id:
                        raise ValueError(
                            f"Negated precondition refers to unknown proposition: {neg_p}"
                        )
                    neg_name = f"not_{neg_p}"
                    if neg_name not in prop_to_id:
                        propositions.append(neg_name)
                        prop_to_id[neg_name] = len(propositions) - 1
                    act["preconds"] = act.get("preconds", []) + [neg_name]
                del act["neg_preconds"]

        for act in actions:
            adds_to_add: List[str] = []
            dels_to_add: List[str] = []
            for add_p in act.get("add", []):
                neg_name = f"not_{add_p}"
                if neg_name in prop_to_id:
                    dels_to_add.append(neg_name)
            for del_p in act.get("del", []):
                neg_name = f"not_{del_p}"
                if neg_name in prop_to_id:
                    adds_to_add.append(neg_name)
            act["add"] = act.get("add", []) + adds_to_add
            act["del"] = act.get("del", []) + dels_to_add

        for p in list(prop_to_id.keys()):
            if p.startswith("not_"):
                orig_p = p[4:]
                if orig_p in prop_to_id and orig_p not in initial:
                    initial.append(p)

        num_props = len(propositions)
        return propositions, prop_to_id, num_props, initial

    def _preprocess_yaml(self) -> None:
        """Load a YAML planning problem and prune unreachable or irrelevant parts."""

        # pylint: disable=too-many-branches,too-many-statements
        with open(self.filename, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        propositions: List[str] = data["propositions"]
        prop_to_id: Dict[str, int] = {p: i for i, p in enumerate(propositions)}

        actions: List[Dict[str, List[str]]] = data["actions"]
        initial: List[str] = data["initial"]

        self._handle_variables(
            actions, propositions, prop_to_id, data.get("variables", {})
        )

        (
            propositions,
            prop_to_id,
            num_props,
            initial,
        ) = self._handle_negated_preconditions(
            actions, propositions, prop_to_id, initial
        )

        noop_actions = []
        for i in range(num_props):
            p = propositions[i]
            noop = {"name": f"noop_{p}", "preconds": [p], "add": [p], "del": []}
            noop_actions.append(noop)

        all_actions = actions + noop_actions
        num_actions = len(all_actions)

        preconds = torch.zeros(
            (num_actions, num_props), dtype=torch.bool, device=self.device
        )
        adds = torch.zeros(
            (num_actions, num_props), dtype=torch.bool, device=self.device
        )
        dels = torch.zeros(
            (num_actions, num_props), dtype=torch.bool, device=self.device
        )

        for idx, act in enumerate(all_actions):
            for p in act.get("preconds", []):
                preconds[idx, prop_to_id[p]] = True
            for p in act.get("add", []):
                adds[idx, prop_to_id[p]] = True
            for p in act.get("del", []):
                dels[idx, prop_to_id[p]] = True

        init_props = torch.zeros(num_props, dtype=torch.bool, device=self.device)
        for p in initial:
            init_props[prop_to_id[p]] = True

        goal_props = torch.zeros(num_props, dtype=torch.bool, device=self.device)
        for p in data["goals"]:
            goal_props[prop_to_id[p]] = True

        forward_prop_mask, forward_act_mask = self._forward_pruning(
            preconds, adds, init_props, num_actions
        )

        if not torch.all(goal_props <= forward_prop_mask):
            print("Goals unreachable from initial state, no solution")
            sys.exit(0)

        reachable_prop_ids = torch.nonzero(forward_prop_mask).squeeze(-1)
        num_reach_props = reachable_prop_ids.size(0)
        if num_reach_props == 0:
            print("No reachable propositions")
            sys.exit(0)

        old_to_new_prop_reach = -torch.ones(
            num_props, dtype=torch.long, device=self.device
        )
        old_to_new_prop_reach[reachable_prop_ids] = torch.arange(
            num_reach_props, device=self.device
        )

        reachable_act_ids = torch.nonzero(forward_act_mask).squeeze(-1)
        num_reach_acts = reachable_act_ids.size(0)

        preconds_reach = preconds[reachable_act_ids][:, reachable_prop_ids]
        adds_reach = adds[reachable_act_ids][:, reachable_prop_ids]
        dels_reach = dels[reachable_act_ids][:, reachable_prop_ids]

        init_reach = init_props[reachable_prop_ids]
        goal_reach = goal_props[reachable_prop_ids]

        all_actions_reach = [all_actions[i.item()] for i in reachable_act_ids]

        backward_prop_mask, backward_act_mask = self._backward_pruning(
            preconds_reach, adds_reach, dels_reach, goal_reach, num_reach_acts
        )

        relevant_prop_ids = torch.nonzero(backward_prop_mask).squeeze(-1)
        num_rel_props = relevant_prop_ids.size(0)
        if num_rel_props == 0:
            print("No relevant propositions")
            sys.exit(0)

        old_to_new_prop = -torch.ones(
            num_reach_props, dtype=torch.long, device=self.device
        )
        old_to_new_prop[relevant_prop_ids] = torch.arange(
            num_rel_props, device=self.device
        )

        relevant_act_ids = torch.nonzero(backward_act_mask).squeeze(-1)
        num_rel_acts = relevant_act_ids.size(0)
        if num_rel_acts == 0 and torch.any(goal_reach & ~init_reach):
            print("No relevant actions to achieve non-initial goals, no solution")
            sys.exit(0)

        preconds_rel = preconds_reach[relevant_act_ids][:, relevant_prop_ids]
        adds_rel = adds_reach[relevant_act_ids][:, relevant_prop_ids]
        dels_rel = dels_reach[relevant_act_ids][:, relevant_prop_ids]

        init_rel = init_reach[relevant_prop_ids]
        goal_rel = goal_reach[relevant_prop_ids]

        self.all_actions = [all_actions_reach[i.item()] for i in relevant_act_ids]
        self.preconds = preconds_rel
        self.adds = adds_rel
        self.dels = dels_rel
        self.init_props = init_rel
        self.goal_props = goal_rel
        self.num_props = num_rel_props

        self.preconds_cpu = self.preconds.cpu()
        self.adds_cpu = self.adds.cpu()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _compute_action_mutex(
        self,
        preconds: torch.Tensor,
        adds: torch.Tensor,
        dels: torch.Tensor,
        prop_mutex: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mutex relationships between applicable actions."""

        pre_f = preconds.float()
        prop_mux_f = prop_mutex.float()
        competing = (pre_f @ prop_mux_f @ pre_f.T) > 0

        adds_a = adds[:, None, :]
        adds_b = adds[None, :, :]
        dels_a = dels[:, None, :]
        dels_b = dels[None, :, :]

        inc_effects = torch.any(adds_a & dels_b, dim=2) | torch.any(
            adds_b & dels_a, dim=2
        )

        pre_a = preconds[:, None, :]
        pre_b = preconds[None, :, :]
        interference = torch.any(dels_a & pre_b, dim=2) | torch.any(
            dels_b & pre_a, dim=2
        )

        mutex = competing | inc_effects | interference
        mutex.diagonal().fill_(False)
        return mutex

    @staticmethod
    def _compute_prop_level(
        adds: torch.Tensor, action_mutex: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the next proposition layer and its mutex matrix."""

        next_props = torch.any(adds, dim=0)
        ach_f = adds.float()
        not_act_mux_f = (~action_mutex).float()
        prop_comp = (ach_f.T @ not_act_mux_f @ ach_f) > 0
        next_prop_mutex = ~prop_comp
        next_prop_mutex.diagonal().fill_(False)
        return next_props, next_prop_mutex

    @staticmethod
    def _find_covering_set(
        level: int,
        subgoals: torch.Tensor,
        achievers: torch.Tensor,
        action_mutex: torch.Tensor,
        failed_cache: Dict[tuple[int, frozenset[int]], bool],
        used: Optional[List[int]] = None,
    ) -> Optional[List[int]]:
        """Return indices of actions covering ``subgoals`` without mutex conflicts."""

        if used is None:
            used = []
        subkey = frozenset(torch.where(subgoals)[0].tolist())
        cache_key = (level, subkey)
        if cache_key in failed_cache:
            return None
        if not torch.any(subgoals):
            return used
        covered = torch.sum(achievers & subgoals[None, :], dim=1)
        possible_acts = torch.argsort(covered, descending=True)
        for act_local in possible_acts:
            if covered[act_local] == 0:
                continue
            if any(action_mutex[act_local, u] for u in used):
                continue
            new_subgoals = subgoals & ~achievers[act_local]
            result = GraphPlan._find_covering_set(
                level,
                new_subgoals,
                achievers,
                action_mutex,
                failed_cache,
                used + [act_local.item()],
            )
            if result is not None:
                return result
        failed_cache[cache_key] = True
        return None

    def _extract_plan(
        self,
        prop_levels: List[Dict[str, torch.Tensor]],
        act_levels: List[Dict[str, torch.Tensor]],
        goal_props: torch.Tensor,
        level: int,
        failed_cache: Dict[tuple[int, frozenset[int]], bool],
    ) -> Optional[List[List[str]]]:
        """Extract a valid plan from the planning graph if possible."""
        current_subgoals = goal_props.clone().cpu()
        plan: List[List[str]] = []

        for lv in range(level, 0, -1):
            act_lv = act_levels[lv - 1]
            act_indices = act_lv["indices"].cpu()
            action_mutex_app = act_lv["mutex"].cpu()
            achievers_app = self.adds_cpu[act_indices]

            selected_local = self._find_covering_set(
                lv, current_subgoals, achievers_app, action_mutex_app, failed_cache
            )
            if selected_local is None:
                return None

            selected_acts = act_indices[selected_local]
            plan_layer = [self.all_actions[i.item()]["name"] for i in selected_acts]
            plan.append(plan_layer)

            current_subgoals = torch.any(self.preconds_cpu[selected_acts], dim=0)

        if torch.all(current_subgoals <= prop_levels[0]["props"].cpu()):
            plan.reverse()
            return plan
        return None

    def run(self) -> Optional[List[List[str]]]:
        """Run the main GraphPlan algorithm and return the plan if found."""

        if torch.all(self.goal_props <= self.init_props):
            return []

        prop_levels = [
            {
                "props": self.init_props,
                "mutex": torch.zeros(
                    (self.num_props, self.num_props),
                    dtype=torch.bool,
                    device=self.device,
                ),
            }
        ]
        act_levels: List[Dict[str, torch.Tensor]] = []

        failed_cache: Dict[tuple[int, frozenset[int]], bool] = {}

        level = 0
        while True:
            level += 1
            current_props = prop_levels[-1]["props"]
            current_prop_mutex = prop_levels[-1]["mutex"]

            applicable = torch.all(self.preconds <= current_props[None, :], dim=1)
            act_indices = torch.where(applicable)[0]
            num_app = act_indices.shape[0]
            if num_app == 0:
                return None

            pre_app = self.preconds[act_indices]
            adds_app = self.adds[act_indices]
            dels_app = self.dels[act_indices]

            action_mutex_app = self._compute_action_mutex(
                pre_app, adds_app, dels_app, current_prop_mutex
            )

            act_levels.append({"indices": act_indices, "mutex": action_mutex_app})

            next_props, next_prop_mutex = self._compute_prop_level(
                adds_app, action_mutex_app
            )

            leveled_off = torch.all(next_props == current_props) and torch.all(
                next_prop_mutex == current_prop_mutex
            )
            if leveled_off:
                return None

            prop_levels.append({"props": next_props, "mutex": next_prop_mutex})

            if torch.all(self.goal_props <= next_props):
                goal_indices = torch.where(self.goal_props)[0]
                if not next_prop_mutex[
                    goal_indices[:, None], goal_indices[None, :]
                ].any():
                    plan = self._extract_plan(
                        prop_levels, act_levels, self.goal_props, level, failed_cache
                    )
                    if plan:
                        return plan

    # ------------------------------------------------------------------
    # Convenience wrapper for command-line usage
    # ------------------------------------------------------------------
    def pretty_print(self, plan: Optional[List[List[str]]]) -> None:
        """Print the plan to stdout in a human readable way."""

        if plan is None:
            print("No plan found")
            return
        if not plan:
            print("Plan found: (empty plan)")
            return
        print("Plan found:")
        for i, layer in enumerate(plan):
            print(f"Step {i + 1}: {layer}")


def main(yaml_file: str, device: Optional[torch.device] = None) -> None:
    """Run GraphPlan on a YAML file and print the resulting plan."""

    planner = GraphPlan(yaml_file, device=device)
    plan = planner.run()
    planner.pretty_print(plan)


if __name__ == "__main__":
    main(sys.argv[1])
