To set up the environment run:

```shell
/bin/bash ./setup.sh
```

To create a zip archive for submission run:

```shell
zip -r submission_assignment_1_DanielLevin___.zip ./ -x "venv/*" -x "*.lprof" -x "instances/*" -x "*.zip" -x ".git/*" -x ".idea/*" -x "*__pycache__/*"
```