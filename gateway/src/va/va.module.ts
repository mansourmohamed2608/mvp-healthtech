import { Module } from '@nestjs/common';
import { VaBookingService } from './va_booking.service';
import { VaController } from './va.controller';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [ConfigModule],
  providers: [VaBookingService],
  controllers: [VaController],
  exports: [VaBookingService],
})
export class VaModule {}
