export class Pool {
  query = jest.fn().mockResolvedValue({ rows: [] });
  connect = jest.fn().mockResolvedValue({
    query: jest.fn().mockResolvedValue({ rows: [] }),
    release: jest.fn(),
  });
  end = jest.fn();
}
