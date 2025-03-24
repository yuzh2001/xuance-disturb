import xuance

if __name__ == "__main__":  # 新增保护块
    runner = xuance.get_runner(
        method="mappo",
        env="mpe",
        env_id="simple_spread_v3",
        is_test=False,
    )
    runner.run()

    import requests

    requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/xuance训练完成")
