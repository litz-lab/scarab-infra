import argparse
from .utilities import prepare_docker_image
from .image_identity import image_tag_for

def main():
    print("BEGIN prepare_docker_image")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-prefix", required=True)
    args = parser.parse_args()
    docker_prefix = args.docker_prefix
    # The tag is derived from the content of this checkout rather than passed
    # in: the generated sbatch script cd's into the infra dir before calling us,
    # so the node computes exactly the hash the submitting host did.
    image_tag = image_tag_for(docker_prefix)
    # Run the image preparation on the local node only
    try:
        prepare_docker_image(
            docker_prefix,
            image_tag
        )
    except:
        print("FAILED prepare_docker_image")
        exit(1)

    print("END prepare_docker_image")

if __name__ == "__main__":
    main()
