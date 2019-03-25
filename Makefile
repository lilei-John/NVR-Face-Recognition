all:
	cd video-face-capture; \
	make -j16; \
	cd -; \

clean:
	cd video-face-capture; \
	make clean; \
	cd -; \