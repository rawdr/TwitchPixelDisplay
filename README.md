# TwitchPixelDisplay
A python script which will update an WLED matrix based on twitch chat messages in a specific format.

Run script to create a blank config.json. In this you will set the IP for your WLED display, what channel's chat to watch, and other configurables.

Right now you can read any channel anonymously using a justinfan#### account, but if this changes in the future there are nick and oauth fields for authenticating.

There is a simple web server at port 9370 where you can see what's on the display.

# Notes
My display uses 4x 32x8 panels in a specific configuration for 64wx16h pixels and the bottom left is considered the origin. Your WLED instance may differ and some code may need to be adjusted.
