import argparse
import os
import shutil

def deploy(model_path, deploy_path):
    os.makedirs(deploy_path, exist_ok=True)
    shutil.copy(os.path.join(model_path, 'model.pth'), os.path.join(deploy_path, 'model_final.pth'))
    print(f"Model deployed to {deploy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--deploy_path', type=str, required=True)
    args = parser.parse_args()
    deploy(args.model_path, args.deploy_path)