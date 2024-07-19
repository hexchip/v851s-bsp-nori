CXXFLAGS += -MMD -MP

INCLUDES += -Iinclude
INCLUDES += -I$(STAGING_DIR)/usr/include/opencv4

LIBS     += -lpthread -ldl -lrt -lVIPlite -lopencv_videoio -lopencv_imgcodecs -lopencv_ml -lopencv_imgproc -lopencv_dnn -lopencv_core

BIN=yolov8

SRCS += ${wildcard *.cpp}
OBJS := $(addsuffix .o, $(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

# Rules

-include $(DEPS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) ${LIBS} -o $@

all: $(BIN) 

clean:
	rm -f $(BIN) $(OBJS) $(DEPS)

.PHONY: all clean