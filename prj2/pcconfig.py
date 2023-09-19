import pynecone as pc

class PrjConfig(pc.Config):
    pass

config = PrjConfig(
    app_name="prj2",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)