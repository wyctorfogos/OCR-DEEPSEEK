Create a .env file:

Then change the input and the output file folder path:
```
    BASE_IN_DIR = Path("./data/to_process")
    BASE_OUT_DIR = Path("./data/output")
```

Docker compose:

To start the services up, go to the Docker folder and (in the terminal) write:
```bash
  docker compose up -d
```

The OCR service will create the JSON file containing the files' content
