"""GraphPlan algorithm implementation using PyTorch tensors."""

import sys

import torch
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_pruning(preconds, adds, init_props, num_actions, device):
    """Perform forward pruning of unreachable propositions and actions."""
    prop_mask = init_props.clone()
    act_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
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


def backward_pruning(preconds, adds, dels, goal_props, num_actions, device):
    """Perform backward pruning to keep only relevant propositions and actions."""
    prop_mask = goal_props.clone()
    act_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
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


def handle_variables(actions, propositions, prop_to_id, variables):
    """Expand variable definitions into propositions and effects."""
    # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
    if not variables:
        return

    var_to_eqs = {}
    var_to_values = {}
    var_defs = {}

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


def handle_negated_preconditions(actions, propositions, prop_to_id, initial):
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
        adds_to_add = []
        dels_to_add = []
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


def preprocess_yaml(yaml_file):
    """Load a YAML planning problem and prune unreachable or irrelevant parts."""
    # pylint: disable=too-many-branches,too-many-statements
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    propositions = data["propositions"]
    prop_to_id = {p: i for i, p in enumerate(propositions)}

    actions = data["actions"]
    initial = data["initial"]

    handle_variables(actions, propositions, prop_to_id, data.get("variables", {}))

    propositions, prop_to_id, num_props, initial = handle_negated_preconditions(
        actions, propositions, prop_to_id, initial
    )

    noop_actions = []
    for i in range(num_props):
        p = propositions[i]
        noop = {"name": f"noop_{p}", "preconds": [p], "add": [p], "del": []}
        noop_actions.append(noop)

    all_actions = actions + noop_actions
    num_actions = len(all_actions)

    preconds = torch.zeros((num_actions, num_props), dtype=torch.bool, device=DEVICE)
    adds = torch.zeros((num_actions, num_props), dtype=torch.bool, device=DEVICE)
    dels = torch.zeros((num_actions, num_props), dtype=torch.bool, device=DEVICE)

    for idx, act in enumerate(all_actions):
        for p in act.get("preconds", []):
            preconds[idx, prop_to_id[p]] = True
        for p in act.get("add", []):
            adds[idx, prop_to_id[p]] = True
        for p in act.get("del", []):
            dels[idx, prop_to_id[p]] = True

    init_props = torch.zeros(num_props, dtype=torch.bool, device=DEVICE)
    for p in initial:
        init_props[prop_to_id[p]] = True

    goal_props = torch.zeros(num_props, dtype=torch.bool, device=DEVICE)
    for p in data["goals"]:
        goal_props[prop_to_id[p]] = True

    forward_prop_mask, forward_act_mask = forward_pruning(
        preconds, adds, init_props, num_actions, DEVICE
    )

    if not torch.all(goal_props <= forward_prop_mask):
        print("Goals unreachable from initial state, no solution")
        sys.exit(0)

    reachable_prop_ids = torch.nonzero(forward_prop_mask).squeeze(-1)
    num_reach_props = reachable_prop_ids.size(0)
    if num_reach_props == 0:
        print("No reachable propositions")
        sys.exit(0)

    old_to_new_prop_reach = -torch.ones(num_props, dtype=torch.long, device=DEVICE)
    old_to_new_prop_reach[reachable_prop_ids] = torch.arange(
        num_reach_props, device=DEVICE
    )

    reachable_act_ids = torch.nonzero(forward_act_mask).squeeze(-1)
    num_reach_acts = reachable_act_ids.size(0)

    preconds_reach = preconds[reachable_act_ids][:, reachable_prop_ids]
    adds_reach = adds[reachable_act_ids][:, reachable_prop_ids]
    dels_reach = dels[reachable_act_ids][:, reachable_prop_ids]

    init_reach = init_props[reachable_prop_ids]
    goal_reach = goal_props[reachable_prop_ids]

    all_actions_reach = [all_actions[i.item()] for i in reachable_act_ids]

    backward_prop_mask, backward_act_mask = backward_pruning(
        preconds_reach, adds_reach, dels_reach, goal_reach, num_reach_acts, DEVICE
    )

    relevant_prop_ids = torch.nonzero(backward_prop_mask).squeeze(-1)
    num_rel_props = relevant_prop_ids.size(0)
    if num_rel_props == 0:
        print("No relevant propositions")
        sys.exit(0)

    old_to_new_prop = -torch.ones(num_reach_props, dtype=torch.long, device=DEVICE)
    old_to_new_prop[relevant_prop_ids] = torch.arange(num_rel_props, device=DEVICE)

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

    all_actions_rel = [all_actions_reach[i.item()] for i in relevant_act_ids]

    return {
        "all_actions": all_actions_rel,
        "preconds": preconds_rel,
        "adds": adds_rel,
        "dels": dels_rel,
        "init_props": init_rel,
        "goal_props": goal_rel,
        "num_props": num_rel_props,
    }


def main(yaml_file):
    """Run GraphPlan on a YAML planning problem and print the resulting plan."""
    # pylint: disable=too-many-branches,too-many-statements
    data = preprocess_yaml(yaml_file)
    all_actions = data["all_actions"]
    preconds = data["preconds"]
    adds = data["adds"]
    dels = data["dels"]
    init_props = data["init_props"]
    goal_props = data["goal_props"]
    num_props = data["num_props"]
    device = preconds.device

    preconds_cpu = preconds.cpu()
    adds_cpu = adds.cpu()

    if torch.all(goal_props <= init_props):
        print("Goals already true in initial state.")
        print("Plan found: (empty plan)")
        return

    prop_levels = [
        {
            "props": init_props,
            "mutex": torch.zeros(
                (num_props, num_props), dtype=torch.bool, device=device
            ),
        }
    ]
    act_levels = []

    failed_cache = {}

    level = 0
    while True:
        level += 1
        current_props = prop_levels[-1]["props"]
        current_prop_mutex = prop_levels[-1]["mutex"]

        applicable = torch.all(preconds <= current_props[None, :], dim=1)
        act_indices = torch.where(applicable)[0]
        num_app = act_indices.shape[0]
        if num_app == 0:
            print("No applicable actions, no solution")
            return

        pre_app = preconds[act_indices]
        adds_app = adds[act_indices]
        dels_app = dels[act_indices]

        pre_app_f = pre_app.float()
        prop_mux_f = current_prop_mutex.float()
        competing_app = (pre_app_f @ prop_mux_f @ pre_app_f.T) > 0

        adds_app_b = adds_app[:, None, :]
        dels_app_b = dels_app[None, :, :]
        inc_effects_app = torch.any(adds_app_b & dels_app_b, dim=2) | torch.any(
            adds_app[None, :, :] & dels_app[:, None, :], dim=2
        )

        dels_app_b = dels_app[:, None, :]
        pre_app_b = pre_app[None, :, :]
        interf_app = torch.any(dels_app_b & pre_app_b, dim=2) | torch.any(
            dels_app[None, :, :] & pre_app[:, None, :], dim=2
        )

        action_mutex_app = competing_app | inc_effects_app | interf_app
        action_mutex_app.diagonal().fill_(False)

        act_levels.append({"indices": act_indices, "mutex": action_mutex_app})

        next_props = torch.any(adds_app, dim=0)
        ach_app_f = adds_app.float()
        not_act_mux_app_f = (~action_mutex_app).float()
        prop_comp = (ach_app_f.T @ not_act_mux_app_f @ ach_app_f) > 0
        next_prop_mutex = ~prop_comp
        next_prop_mutex.diagonal().fill_(False)

        leveled_off = torch.all(next_props == current_props) and torch.all(
            next_prop_mutex == current_prop_mutex
        )
        if leveled_off:
            print("Graph leveled off, no solution exists")
            return

        prop_levels.append({"props": next_props, "mutex": next_prop_mutex})

        if torch.all(goal_props <= next_props):
            goal_indices = torch.where(goal_props)[0]
            if not next_prop_mutex[goal_indices[:, None], goal_indices[None, :]].any():
                print(f"Goals reachable at level {level}, attempting extraction...")
                plan = extract_plan(
                    prop_levels,
                    act_levels,
                    all_actions,
                    goal_props,
                    level,
                    preconds_cpu,
                    adds_cpu,
                    failed_cache,
                )
                if plan:
                    print("Plan found:")
                    for i, layer in enumerate(plan):
                        print(f"Step {i+1}: {layer}")
                    return


def extract_plan(
    prop_levels,
    act_levels,
    all_actions,
    goal_props,
    level,
    preconds_cpu,
    adds_cpu,
    failed_cache,
):
    """Extract a valid plan from the planning graph if possible."""
    # pylint: disable=cell-var-from-loop
    current_subgoals = goal_props.clone().cpu()
    plan = []

    for lv in range(level, 0, -1):
        act_lv = act_levels[lv - 1]
        act_indices = act_lv["indices"].cpu()
        action_mutex_app = act_lv["mutex"].cpu()
        achievers_app = adds_cpu[act_indices]

        def find_covering_set(subgoals, used=None):
            if used is None:
                used = []
            subkey = frozenset(torch.where(subgoals)[0].tolist())
            cache_key = (lv, subkey)
            if cache_key in failed_cache:
                return None
            if not torch.any(subgoals):
                return used
            covered = torch.sum(achievers_app & subgoals[None, :], dim=1)
            possible_acts = torch.argsort(covered, descending=True)
            for act_local in possible_acts:
                if covered[act_local] == 0:
                    continue
                if any(action_mutex_app[act_local, u] for u in used):
                    continue
                new_subgoals = subgoals & ~achievers_app[act_local]
                result = find_covering_set(new_subgoals, used + [act_local.item()])
                if result is not None:
                    return result
            failed_cache[cache_key] = True
            return None

        selected_local = find_covering_set(current_subgoals)
        if selected_local is None:
            return None

        selected_acts = act_indices[selected_local]
        plan_layer = [all_actions[i.item()]["name"] for i in selected_acts]
        plan.append(plan_layer)

        current_subgoals = torch.any(preconds_cpu[selected_acts], dim=0)

    if torch.all(current_subgoals <= prop_levels[0]["props"].cpu()):
        plan.reverse()
        return plan
    return None


if __name__ == "__main__":
    main(sys.argv[1])
