import os
import yaml
import argparse
from pathlib import Path

def load_cfg(file_path):
    with open(os.path.join(os.getcwd(), file_path), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    return cfg

def copy_cfg(file_path, target_path):
    import subprocess
    Path(target_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", file_path, target_path])

def get_params():
    parser = argparse.ArgumentParser(
        prog="Z1 training",
    )
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=24000)
    parser.add_argument("--control_freq", type=int, default=None) # how frequent low-level request high-level
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=-1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="isaacgym")
    parser.add_argument("--wandb_name", type=str, default="isaacgym")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Weights & Biases entity (username or team). If unset, uses WANDB_ENTITY or logged-in viewer.",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--debugvis", action="store_true")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wrist_seg", action="store_true")
    parser.add_argument("--front_only", action="store_true")
    parser.add_argument("--seperate", action="store_true")
    parser.add_argument("--teacher_ckpt_path", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--roboinfo", action="store_true")
    parser.add_argument("--observe_gait_commands", action="store_true")
    parser.add_argument("--small_value_set_zero", action="store_true")
    parser.add_argument("--fixed_base", action="store_true")
    parser.add_argument("--use_tanh", action="store_true")
    parser.add_argument("--reach_only", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--last_commands", action="store_true")
    parser.add_argument("--no_feature", action="store_true")
    parser.add_argument("--mask_arm", action="store_true")
    parser.add_argument("--mlp_stu", action="store_true")
    parser.add_argument("--depth_random", action="store_true")
    parser.add_argument("--pitch_control", action="store_true")
    parser.add_argument("--pred_success", action="store_true")
    parser.add_argument("--near_goal_stop", action="store_true")
    parser.add_argument("--obj_move_prob", type=float, default=0.0)
    parser.add_argument("--rand_control", action="store_true")
    parser.add_argument("--arm_delay", action="store_true")
    parser.add_argument("--rand_cmd_scale", action="store_true")
    parser.add_argument("--rand_depth_clip", action="store_true")
    parser.add_argument("--stop_pick", action="store_true")
    parser.add_argument("--arm_kp", type=int, default=40) # only useful when log data
    parser.add_argument("--arm_kd", type=float, default=2) # only useful when log data
    parser.add_argument(
        "--table_height",
        type=float,
        default=None,
        help="World z of tabletop upper surface (m); overrides cfg env.table_surface_height.",
    )
    parser.add_argument("--seed", type=int, default=43) # only useful when log data
    
    args = parser.parse_args()
    
    return args


def resolve_wandb_entity(args):
    """Return (entity, source). Source is cli_or_env | api_viewer_team | api_viewer_username | none.

    Prefer team slug over username: W&B ``username`` is often not a valid ``entity`` for upsertBucket
    (e.g. runs must live under a team like ``huangnancy``, not ``huangnancy1122``).
    """
    e = (getattr(args, "wandb_entity", "") or os.environ.get("WANDB_ENTITY") or "").strip()
    if e:
        return e, "cli_or_env"
    try:
        import wandb

        viewer = wandb.Api().viewer
        teams = getattr(viewer, "teams", None) or []
        if teams:
            t0 = teams[0]
            if isinstance(t0, dict):
                t0 = t0.get("name") or t0.get("entity")
            if t0:
                return str(t0).strip(), "api_viewer_team"
        un = getattr(viewer, "username", None)
        if un and str(un).strip():
            return str(un).strip(), "api_viewer_username"
    except Exception:
        pass
    return "", "none"


def build_wandb_kwargs(args):
    """Return (kwargs, entity_source) for skrl wandb.init; entity from CLI/env or W&B viewer."""
    entity, src = resolve_wandb_entity(args)
    kwargs = {"project": args.wandb_project, "tensorboard": False, "name": args.wandb_name}
    if entity:
        kwargs["entity"] = entity
        os.environ["WANDB_ENTITY"] = entity
    return kwargs, src
