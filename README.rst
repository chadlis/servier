=======
servier
=======






Drug molecule properties prediction



Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


Commands
-------

Build image. The folder /data will not be copied (check .dockerignore)
docker build . -t servier

Run docker container with data as a mounted volume, and run a local bash
docker run --rm -it --mount type=bind,source="/Users/chadli/Documents/projects/servier/data",target=/app/data --entrypoint bash servier

Split data
servier split
