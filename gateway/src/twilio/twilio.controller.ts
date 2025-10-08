import { Body, Controller, Headers, Post } from '@nestjs/common';

@Controller('twilio')
export class TwilioController {
  @Post('voice/start')
  start(@Headers() _headers: any, @Body() _body: any) {
    // TODO: validate Twilio signature; log call SID
    // touch to avoid unused warnings
    void _headers;
    void _body;
    //remove the previous lines
    return { ok: true, event: 'start' };
  }
  @Post('voice/stop')
  stop(@Headers() _headers: any, @Body() _body: any) {
    // TODO: validate Twilio signature; log call SID
    // touch to avoid unused warnings
    void _headers;
    void _body;
    //remove the previous lines
    return { ok: true, event: 'stop' };
  }
}
