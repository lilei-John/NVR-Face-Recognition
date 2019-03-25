#SDK测试程序

CXX   =  g++ -std=c++11
CXXFLAGS = -Wall -Wno-strict-aliasing -Wno-unused-variable

#包含头文件路径
SUBDIR   = $(shell ls ./src -R | grep /)
SUBDIRS  = $(subst :,/,$(SUBDIR))
INCPATHS = $(subst ./,-I./,$(SUBDIRS))
INCPATHS += -I./include/
INCPATHS += -I./include/seeta/


VPATH = $(subst : ,:,$(SUBDIR))./
SOURCE = $(foreach dir,$(SUBDIRS),$(wildcard $(dir)*.cpp))

#将cpp的路径名称的.cpp后缀换为.o
OBJS = $(patsubst %.cpp,%.o,$(SOURCE))
#取出.o的文件名，不带路径
OBJFILE  = $(foreach dir,$(OBJS),$(notdir $(dir)))
#.o输出到.OBJ目录下
OBJSPATH = $(addprefix obj/,$(OBJFILE)) 
#-lX11 编译时指定动态库的搜索路径, 在系统里添加LD_LIBRARY_PATH
#LIBPATH = ./bin/lib/ #-Wl,-rpath=./:$(LIBPATH)HCNetSDKCom:$(LIBPATH)
LIBS = -lHCPreview -lhcnetsdk -lseeta_fa_lib -lX11 -lPlayCtrl -lpthread -lseeta_facedet_lib `pkg-config --cflags --libs opencv`
EXE = ./bin/nvr_face_capture

$(EXE):$(OBJFILE)
	$(CXX) -L$(LIBPATH)  -o $(EXE) $(OBJFILE) $(INCPATHS) $(LIBS)

$(OBJFILE):%.o:%.cpp
	$(CXX) -c -o $@ $<  $(INCPATHS) -pipe -g -Wall `pkg-config --cflags --libs opencv`

#依赖文件
DPPS = $(patsubst %.cpp,%.dpp,$(SOURCE))
include $(DPPS)
%.dpp: %.cpp	
	$(CXX) $(INCPATHS) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$ 
	

.PHONY:clean
clean:
	rm -rf $(OBJFILE)
	rm -rf $(DPPS)
	rm -rf $(EXE)

