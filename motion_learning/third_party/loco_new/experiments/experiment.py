from rl_x.runner.runner import Runner


if __name__ == "__main__":
    runner = Runner(implementation_package_names=["rl_x", "loco_new"])
    runner.run()