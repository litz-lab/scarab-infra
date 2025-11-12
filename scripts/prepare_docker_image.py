import argparse
from .utilities import prepare_docker_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-prefix", required=True)
    parser.add_argument("--githash", required=True)
    args = parser.parse_args()
    docker_prefix = args.docker_prefix
    githash = args.githash
    image_tag = f"{docker_prefix}:{githash}"
    # Run the image preparation on the local node only
    prepare_docker_image(
        docker_prefix,
        image_tag
    )

if __name__ == "__main__":
    main()
