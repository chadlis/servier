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

Download Tensorflow wheen
wget https://tf.novaal.de/barcelona/tensorflow-2.6.0-cp38-cp38-linux_x86_64.whl
wget https://tf.novaal.de/barcelona/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
(https://github.com/yaroslavvb/tensorflow-community-wheels/issues/198)

Build image. The folder /data will not be copied (check .dockerignore)
servier % docker build . --platform linux/amd64 -t servier

Run docker container with data as a mounted volume, and run a local bash
docker run --rm -it --memory="8g" --cpus="8" --mount type=bind,source="/Users/chadli/Documents/projects/servier/data",target=/app/data --platform linux/amd64 --entrypoint bash servier

Split data
servier split


Remove image
docker image rm servier


RUN pip install keras==2.6