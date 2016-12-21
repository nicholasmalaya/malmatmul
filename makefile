## Select build enviroment  ##

#CC=icpc #intel
CC=g++  #GNU
CC=hcc

CFLAGS  = #-03 #-g
CXXFLAGS=-Wall 
LDFLAGS = 
SOURCES = malmatmul.cpp 

## executable name ##
EXEC=malmatmul

###############################################
######### DO not edit anything below ##########
###############################################                              

## append include and library paths -- choose what is being used here ##
INCLUDE=
LIBS=

## building object list ##
OBJECTS = $(addsuffix .o,$(basename $(SOURCES)))

## build the executable ##
$(EXEC): $(OBJECTS)
	$(CC) `hcc-config --cxxflags --ldflags` $(LDFLAGS) $(OBJECTS) -o $(INCLUDE) $@ $(LIBS) 

# build cpp object files
%.o: %.cpp
	@echo building $< 
	$(CC) `hcc-config --cxxflags --ldflags` -c $(CXXFLAGS) $(INCLUDE) $< -o $@

# build cpp object files
%.o: %.c
	@echo building $< 
	$(CC)  -c $(CFLAGS) $(INCLUDE) $< -o $@

# clean directive 'make clean'
clean:
	- /bin/rm $(EXEC) *.o *.mod *~ \#* 
	@echo 'files cleaned'
