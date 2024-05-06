#include "sys_utils.h"

#import "cocoa/cocoa.h"

namespace cmp
{


std::string getCurrFilePath()
{
    @autoreleasepool {
        NSString *currentpath = [[NSBundle mainBundle] bundlePath];
        
        return std::string([currentpath UTF8String]) + "/Contents";
    }
}

}
