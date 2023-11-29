# An alternative way to build opensbi for RISC-V

`docker-compose up` is a command that is used to start and run multi-container Docker applications.
It reads the `docker-compose.yml` file and starts the services defined in it.
This command is useful for managing complex applications that require multiple containers.
In this case we are creating services to build the two binaries: linux and opensbi.

On the other hand, `docker buildx` is a Docker CLI plugin that extends the `docker build` command with the ability to build images for multiple platforms.
It allows you to create and manage builder instances, which are used to build images for different architectures and operating systems.
This is useful for building images that can run on different platforms, such as ARM and x86. In this case we use it to build to RISC-V.

When doing multiplatform builds, `docker-compose up` can still be used to manage the containers, but `docker buildx` must be used to build the images for multiple platforms.
This involves creating a builder instance with `docker buildx create`, setting it as the default builder with `docker buildx use`, and then running `docker buildx build` with the `--platform` flag to specify the target platforms.

This example is not complete but should provide a good example of how this could be done.
I cannot get opensbi to build, but this is mostly due to not knowing how to setup the correct environment in "Dockerfile-opensbi" (my build command in "docker-compose.yaml" may also be incorrect).

Once complete all a user would need to do is run `docker-compose up` and the two binaries would be built and available for use.
