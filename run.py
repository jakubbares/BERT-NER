from model_runner import ModelRunner

runner = ModelRunner()
runner.train_and_save()
runner.load_model()
runner.test()


