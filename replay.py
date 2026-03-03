from __future__ import annotations

import argparse
import sys

import pygame
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from pvz_env import ACTION_MEANINGS, PvZEnv, ScriptedBaselinePolicy
from pvz_env import config


def action_mask_fn(env: PvZEnv):
    return env.get_action_mask()

WIDTH, HEIGHT = 1280, 760
CELL_W, CELL_H = 120, 120
MARGIN_X, MARGIN_Y = 80, 40


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
    p.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--stochastic", action="store_true", help="sample actions for stochastic policy playback")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = PvZEnv(seed=args.seed, difficulty=args.difficulty)
    if args.algo == "maskable":
        env = ActionMasker(env, action_mask_fn)
    obs, info = env.reset(seed=args.seed)

    if args.policy == "ppo":
        model = MaskablePPO.load(args.model) if args.algo == "maskable" else PPO.load(args.model)
    else:
        model = None
    scripted = ScriptedBaselinePolicy() if args.policy == "scripted" else None

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PvZ RL Replay")
    font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()

    done = False
    last_action = 0
    ep_reward = 0.0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        if args.policy == "ppo":
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action = int(action)
        elif args.policy == "random":
            action = env.action_space.sample()
        else:
            action = int(scripted.predict(obs))

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        ep_reward += reward
        last_action = action
        draw(screen, font, info["snapshot"], last_action, ep_reward, info.get("sun_collected", 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.time.wait(1200)
    pygame.quit()


def draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    snapshot: dict,
    action: int,
    ep_reward: float,
    last_collected: int,
) -> None:
    screen.fill(COLORS["bg"])
    for lane in range(config.LANES):
        for col in range(config.COLS):
            x = MARGIN_X + col * CELL_W
            y = MARGIN_Y + lane * CELL_H
            pygame.draw.rect(screen, COLORS["grid"], pygame.Rect(x, y, CELL_W - 2, CELL_H - 2), 2)

    for p in snapshot["plants"]:
        x = MARGIN_X + p["col"] * CELL_W + CELL_W // 2
        y = MARGIN_Y + p["lane"] * CELL_H + CELL_H // 2
        color = COLORS[p["kind"]]
        pygame.draw.circle(screen, color, (x, y), 24)

    for z in snapshot["zombies"]:
        x = MARGIN_X + int(z["x"] * CELL_W)
        y = MARGIN_Y + z["lane"] * CELL_H + CELL_H // 2
        pygame.draw.rect(screen, COLORS[z["kind"]], pygame.Rect(x - 16, y - 24, 28, 48))

    for s in snapshot["loose_sun"]:
        x = MARGIN_X + int(s["x"] * CELL_W)
        y = MARGIN_Y + s["lane"] * CELL_H + 20
        pygame.draw.circle(screen, COLORS["sun"], (x, y), 10)

    text = (
        f"Step={snapshot['step']}  Sun={snapshot['sun']}  Action={ACTION_MEANINGS[action]}  "
        f"Reward={ep_reward:.2f}  Mowers={snapshot['mowers']}  "
        f"LooseSun={len(snapshot['loose_sun'])}  Sunflowers={snapshot.get('sunflowers_total', 0)}  "
        f"LastCollected={last_collected}"
    )
    screen.blit(font.render(text, True, COLORS["text"]), (20, HEIGHT - 40))


if __name__ == "__main__":
    main()
