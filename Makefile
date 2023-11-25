setup:
	python3 -m venv venv/ && . venv/bin/activate
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

install_data:
	rm -rf data/digital_leaders/* && \
    cd data/digital_leaders && wget https://storage.yandexcloud.net/datapub/train_updated_titles.zip && \
    unzip train_updated_titles.zip
	mkdir -p masks/ images/
	ls
	cp -r train_updated_titiles/masks/ ./masks/ && cp -r train_updated_titiles/images/ ./images/
	rm -R train_updated_titiles && rm train_updated_titles.zip && rm __MACOSX/


generate_tiles:
	cd src/preprocessing && python tile_generating.py