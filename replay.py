from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pygame
from pvz_env import ACTION_MEANINGS, PvZEnv, ScriptedBaselinePolicy
from pvz_env import config
from pvz_env.sprite_loader import safe_load_png, scale_surface

WIDTH, HEIGHT = 1280, 760
BOARD_TOP = 110
BOARD_BOTTOM = 30
BOARD_LEFT = 70
BOARD_RIGHT = 70

COLORS = {
    "bg": (34, 120, 45),
    "grid": (190, 220, 170),
    "sunflower": (244, 219, 79),
    "peashooter": (80, 205, 100),
    "wallnut": (139, 94, 60),
    "normal": (120, 120, 120),
    "conehead": (160, 110, 80),
    "sun": (255, 215, 80),
    "text": (250, 250, 250),
}



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pygame replay of one episode")
    p.add_argument("--model", type=str, default="models/ppo_pvz_final.zip")
    p.add_argument("--policy", choices=["ppo", "random", "scripted"], default="ppo")
    p.add_argument("--algo", choices=["ppo", "maskable"], default="ppo")
    p.add_argument("--stochastic", action="store_true", help="sample actions for stochastic policy playback")
    p.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-sprites", action="store_true")
    p.add_argument("--sprite-scale", type=float, default=1.0)
    return p.parse_args()


def safe_font(size: int) -> Optional[pygame.font.Font]:
    try:
        return pygame.font.SysFont("consolas", size)
    except pygame.error:
        try:
            return pygame.font.Font(None, size)
        except pygame.error:
            return None


def handle_quit_events() -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)


def draw_loading(screen: pygame.Surface, font: Optional[pygame.font.Font], line1: str, line2: str) -> None:
    screen.fill((20, 30, 20))
    if font is None:
        pygame.display.flip()
        return
    screen.blit(font.render(line1, True, COLORS["text"]), (40, HEIGHT // 2 - 30))
    screen.blit(font.render(line2, True, COLORS["text"]), (40, HEIGHT // 2 + 10))
    pygame.display.flip()


def compute_layout() -> tuple[int, int, int, int]:
    board_w = WIDTH - BOARD_LEFT - BOARD_RIGHT
    board_h = HEIGHT - BOARD_TOP - BOARD_BOTTOM
    tile_w = board_w // config.COLS
    tile_h = board_h // config.LANES
    board_w = tile_w * config.COLS
    board_h = tile_h * config.LANES
    return board_w, board_h, tile_w, tile_h


def world_to_screen(lane: int, x: float, tile_w: int, tile_h: int) -> tuple[int, int]:
    screen_x = BOARD_LEFT + int(x * tile_w)
    screen_y = BOARD_TOP + lane * tile_h + tile_h // 2
    return screen_x, screen_y


def scale_to_height(surface: pygame.Surface, target_h: int) -> pygame.Surface:
    w, h = surface.get_size()
    ratio = w / max(1, h)
    return scale_surface(surface, int(target_h * ratio), target_h)


def load_sprites(tile_w: int, tile_h: int, sprite_scale: float, disabled: bool) -> dict[str, Optional[pygame.Surface]]:
    if disabled:
        return {k: None for k in ["sunflower", "peashooter", "wallnut", "normal", "conehead", "sun"]}

    assets = Path("assets")
    plant_target_w = int(tile_w * 0.9 * sprite_scale)
    plant_target_h = int(tile_h * 0.9 * sprite_scale)
    zombie_target_h = int(tile_h * 0.95 * sprite_scale)
    sun_target = int(min(tile_w, tile_h) * 0.24 * sprite_scale)

    sprites: dict[str, Optional[pygame.Surface]] = {
        "sunflower": None,
        "peashooter": None,
        "wallnut": None,
        "normal": None,
        "conehead": None,
        "sun": None,
    }

    plant_files = {
        "sunflower": assets / "plants" / "sunflower.png",
        "peashooter": assets / "plants" / "peashooter.png",
        "wallnut": assets / "plants" / "wallnut.png",
    }
    zombie_files = {
        "normal": assets / "zombies" / "zombie.png",
        "conehead": assets / "zombies" / "cone_zombie.png",
    }

    for name, pth in plant_files.items():
        raw = safe_load_png(str(pth))
        if raw is not None:
            sprites[name] = scale_surface(raw, plant_target_w, plant_target_h)

    for name, pth in zombie_files.items():
        raw = safe_load_png(str(pth))
        if raw is not None:
            sprites[name] = scale_to_height(raw, zombie_target_h)

    sun_raw = safe_load_png(str(assets / "ui" / "sun.png"))
    if sun_raw is not None:
        sprites["sun"] = scale_surface(sun_raw, sun_target, sun_target)

    return sprites


def draw(
    screen: pygame.Surface,
    font: Optional[pygame.font.Font],
    snapshot: dict,
    action: int,
    done: bool,
    tile_w: int,
    tile_h: int,
    sprites: dict[str, Optional[pygame.Surface]],
    mask_sum: Optional[int] = None,
    collect_legal: Optional[bool] = None,
) -> None:
    screen.fill(COLORS["bg"])
    for lane in range(config.LANES):
        for col in range(config.COLS):
            x = BOARD_LEFT + col * tile_w
            y = BOARD_TOP + lane * tile_h
            pygame.draw.rect(screen, COLORS["grid"], pygame.Rect(x, y, tile_w - 2, tile_h - 2), 2)

    for plant in snapshot["plants"]:
        center_x, center_y = world_to_screen(plant["lane"], plant["col"] + 0.5, tile_w, tile_h)
        sprite = sprites.get(plant["kind"])
        if sprite is not None:
            rect = sprite.get_rect(center=(center_x, center_y))
            screen.blit(sprite, rect)
        else:
            pygame.draw.circle(screen, COLORS[plant["kind"]], (center_x, center_y), int(min(tile_w, tile_h) * 0.2))

    for zombie in snapshot["zombies"]:
        x, y = world_to_screen(zombie["lane"], zombie["x"], tile_w, tile_h)
        sprite = sprites.get(zombie["kind"])
        if sprite is not None:
            rect = sprite.get_rect(midbottom=(x, y + tile_h // 2 - 4))
            screen.blit(sprite, rect)
        else:
            pygame.draw.rect(screen, COLORS[zombie["kind"]], pygame.Rect(x - 16, y - 24, 28, 48))

    for sun in snapshot["loose_sun"]:
        x, y = world_to_screen(sun["lane"], sun["x"] + 0.5, tile_w, tile_h)
        y = y - tile_h // 3
        sprite = sprites.get("sun")
        if sprite is not None:
            rect = sprite.get_rect(center=(x, y))
            screen.blit(sprite, rect)
        else:
            pygame.draw.circle(screen, COLORS["sun"], (x, y), 10)

    if font is not None:
        lines = [
            f"step: {snapshot['step']}",
            f"action: {ACTION_MEANINGS[action]}",
            f"sun: {snapshot['sun']}",
            f"loose sun: {len(snapshot['loose_sun'])}",
            f"plants: {len(snapshot['plants'])}",
            f"zombies: {len(snapshot['zombies'])}",
        ]
        if mask_sum is not None:
            lines.append(f"mask_sum: {mask_sum}")
        if collect_legal is not None:
            lines.append(f"collect_legal: {collect_legal}")
        if done:
            status = "WIN" if snapshot.get("win") else "LOSS"
            lines.append(f"result: {status}")
        for idx, line in enumerate(lines):
            screen.blit(font.render(line, True, COLORS["text"]), (20, 10 + idx * 24))


def main() -> None:
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PvZ RL Replay")
    font = safe_font(24)
    print(f"[{time.time():.3f}] pygame window created")

    draw_loading(screen, font, "Loading replay...", "Loading model...")
    handle_quit_events()

    model = None
    if args.policy == "ppo":
        model_path = str(args.model).lower()
        if args.algo == "ppo" and "maskable" in model_path:
            print(
                "Warning: model path contains 'maskable' but --algo is 'ppo'. "
                "If this is a MaskablePPO checkpoint, use --algo maskable."
            )

        if args.algo == "maskable":
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(args.model)
        else:
            from stable_baselines3 import PPO

            model = PPO.load(args.model)
        print(f"[{time.time():.3f}] model loaded")

    draw_loading(screen, font, "Loading replay...", "Preparing env...")
    handle_quit_events()

    env = PvZEnv(seed=args.seed, difficulty=args.difficulty)
    if args.algo == "maskable":
        from sb3_contrib.common.wrappers import ActionMasker

        env = ActionMasker(env, lambda e: e.get_action_mask())
    obs, info = env.reset(seed=args.seed)
    print(f"[{time.time():.3f}] env reset")

    scripted = ScriptedBaselinePolicy() if args.policy == "scripted" else None
    _, _, tile_w, tile_h = compute_layout()
    sprites = load_sprites(tile_w, tile_h, args.sprite_scale, args.no_sprites)

    clock = pygame.time.Clock()
    done = False
    last_action = 0
    first_frame_printed = False

    while not done and info["snapshot"]["step"] < args.max_steps:
        mask_sum = None
        collect_legal = None
        handle_quit_events()

        if args.policy == "ppo":
            if args.algo == "maskable":
                base_env = env.unwrapped
                masks = base_env.get_action_mask()
                mask_sum = int(masks.sum())
                collect_legal = bool(masks[1])
                action, _ = model.predict(obs, deterministic=not args.stochastic, action_masks=masks)
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            action = int(action)
        elif args.policy == "random":
            action = env.action_space.sample()
        else:
            action = int(scripted.predict(obs))

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        last_action = action

        draw(
            screen,
            font,
            info["snapshot"],
            last_action,
            done,
            tile_w,
            tile_h,
            sprites,
            mask_sum=mask_sum,
            collect_legal=collect_legal,
        )
        pygame.display.flip()

        if not first_frame_printed:
            print(f"[{time.time():.3f}] first frame rendered")
            first_frame_printed = True
        clock.tick(args.fps)

    pygame.time.wait(500)
    pygame.quit()


if __name__ == "__main__":
    main()
