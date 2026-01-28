import { INestApplicationContext, Logger } from '@nestjs/common';
import { AbstractWsAdapter } from '@nestjs/websockets';
import { MessageMappingProperties } from '@nestjs/websockets';
import { CLOSE_EVENT } from '@nestjs/websockets/constants';
import { Observable, fromEvent, EMPTY } from 'rxjs';
import { filter, first, mergeMap, share, takeUntil } from 'rxjs/operators';
import * as WebSocket from 'ws';

/**
 * Custom WebSocket Adapter for Twilio Media Streams
 * Accepts WebSocket connections on any path (e.g., /twilio/{callSid})
 */
export class TwilioWsAdapter extends AbstractWsAdapter {
  private readonly logger = new Logger(TwilioWsAdapter.name);

  constructor(app: INestApplicationContext) {
    super(app);
  }

  create(port: number, options?: any): any {
    const server = this.httpServer;
    
    const wsServer = new WebSocket.Server({
      noServer: true,
      ...options,
    });

    // Handle upgrade requests for ALL paths
    server.on('upgrade', (request: any, socket: any, head: Buffer) => {
      const pathname = new URL(request.url, `http://${request.headers.host}`).pathname;
      
      this.logger.debug(`WebSocket upgrade request for path: ${pathname}`);
      
      // Accept ALL WebSocket connections - let the gateway filter them
      wsServer.handleUpgrade(request, socket, head, (ws) => {
        wsServer.emit('connection', ws, request);
      });
    });

    return wsServer;
  }

  bindClientConnect(server: WebSocket.Server, callback: Function) {
    server.on('connection', callback);
  }

  bindClientDisconnect(client: WebSocket, callback: Function) {
    client.on('close', callback);
  }

  bindMessageHandlers(
    client: WebSocket,
    handlers: MessageMappingProperties[],
    transform: (data: any) => Observable<any>,
  ) {
    const close$ = fromEvent(client, CLOSE_EVENT).pipe(share(), first());
    const source$ = fromEvent(client, 'message').pipe(
      mergeMap((data: any) =>
        this.bindMessageHandler(data, handlers, transform).pipe(
          filter((result) => result !== undefined),
        ),
      ),
      takeUntil(close$),
    );
    source$.subscribe((response) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(response));
      }
    });
  }

  bindMessageHandler(
    buffer: any,
    handlers: MessageMappingProperties[],
    transform: (data: any) => Observable<any>,
  ): Observable<any> {
    try {
      const message = JSON.parse(buffer.data);
      const messageHandler = handlers.find(
        (handler) => handler.message === message.event,
      );
      if (!messageHandler) {
        return EMPTY;
      }
      return transform(messageHandler.callback(message));
    } catch (error) {
      this.logger.error('Error parsing WebSocket message', error);
      return EMPTY;
    }
  }

  close(server: WebSocket.Server) {
    server.close();
  }
}
